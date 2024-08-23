import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from fuzzywuzzy import process
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

# Fine-tuning model setup
model_name = "NousResearch/Llama-2-7b-chat-hf"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25
device_map = {"": 0}

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
max_seq_length = 512

# Functions from the original Streamlit app
def process_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    df['product_title'] = df['Product Name'].fillna('').astype(str).str.lower().str.strip()
    texts = list(map(lambda x: x.replace("\n", " "), df['product_title'].tolist()))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts, embeddings)
    return df, knowledge_base

def get_response(text, tokenizer=tokenizer, model=model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(model.device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048 - input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return response

def process_query(df, knowledge_base, query):
    docs = knowledge_base.similarity_search(query, k=10)
    results = []
    seen_titles = set()
    doc_titles = [doc.page_content.lower() for doc in docs]
    best_matches = process.extract(query.lower(), doc_titles, limit=10)
    
    for match in best_matches:
        product_title = match[0].strip()
        if product_title in seen_titles:
            continue
        seen_titles.add(product_title)
        
        sku_matches = df.loc[df['product_title'] == product_title, 'SKU'].values
        if len(sku_matches) > 0:
            results.append(f"{product_title.title()}\nSKU: {sku_matches[0]}")
    
    if results:
        return "We have found:\n\n" + "\n\n".join(results)
    else:
        return "No matching products found."

def handle_prompt(prompt, df=None, knowledge_base=None):
    food_keywords = ["recipe", "cook", "bake", "grill", "fry", "boil", "juice", "smoothie", "ingredient", "dish","how to make","generate","give me "]

    if not any(keyword in prompt.lower() for keyword in food_keywords):
        if df is not None and knowledge_base is not None:
            return process_query(df, knowledge_base, prompt)
        return "I apologize, but I'm only able to provide recipes and information related to food and juices."

    expert_prompt = f"You are a master chef and food expert. Only answer with detailed recipes related to food or juices. Do not answer any other queries. {prompt}"
    response = get_response(expert_prompt)

    if not any(keyword in response.lower() for keyword in food_keywords):
        return "I apologize, but I'm only able to provide recipes and information related to food and juices."

    return response

if __name__ == '__main__':
    st.set_page_config(page_title="Chat with Spinneys AI", page_icon=":shark:", layout="wide")
    st.title("Chat with Spinneys AI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload CSV")
        if 'df' not in st.session_state or 'knowledge_base' not in st.session_state:
            csv_file = st.file_uploader("Upload your CSV File", type="csv")
            
            if csv_file is not None:
                with st.spinner("Processing..."):
                    df, knowledge_base = process_csv(csv_file)
                    st.session_state.df = df
                    st.session_state.knowledge_base = knowledge_base
                st.success("CSV successfully uploaded and processed!")
        else:
            st.write("CSV file already uploaded and processed!")

    with col2:
        st.header("Chat")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What product are you looking for?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if 'df' in st.session_state:
                response = handle_prompt(prompt, st.session_state.df, st.session_state.knowledge_base)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = handle_prompt(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
