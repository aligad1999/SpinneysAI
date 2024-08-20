import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.llms import HuggingFaceHub
import fuzzywuzzy
from fuzzywuzzy import process
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    df['product_title'] = df['Product Name'].fillna('').astype(str).str.lower().str.strip()
    texts = list(map(lambda x: x.replace("\n", " "), df['product_title'].tolist()))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts, embeddings)
    return df, knowledge_base

def get_llm():
    api_key = "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"
    return HuggingFaceHub(
        repo_id="gpt2-xl",
        model_kwargs={"temperature": 0.5, "max_length": 1024},
        huggingfacehub_api_token=api_key
    )

def process_query(df, knowledge_base, query, llm):
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

def generate_recipe(prompt, llm):
    formatted_prompt = (
        "You are a professional chef and recipe generator. Your task is to create a detailed, clear, and concise recipe based on the user's request. Please ensure the response is well-structured and includes the following sections without any additional commentary:\n\n"
        "- **Ingredients**: List all necessary ingredients with exact quantities.\n"
        "- **Equipment**: List all required equipment.\n"
        "- **Instructions**: Provide clear, step-by-step cooking instructions.\n"
        "- **Tips**: Offer any useful tips, variations, or serving suggestions.\n\n"
        "Please deliver the recipe in a professional tone, free of any unnecessary content or extraneous formatting instructions. Focus solely on the recipe details.\n\n"
        "Request: {}"
    ).format(prompt)

    response = llm(formatted_prompt)
    sections = ["Ingredients", "Equipment", "Instructions", "Tips"]
    final_response = []

    for section in sections:
        start_index = response.find(f"**{section}**:")
        if start_index != -1:
            end_index = len(response)
            for next_section in sections[sections.index(section) + 1:]:
                next_index = response.find(f"**{next_section}**:", start_index)
                if next_index != -1:
                    end_index = min(end_index, next_index)
            section_content = response[start_index:end_index].strip()
            final_response.append(section_content)

    formatted_response = "\n\n".join(final_response)
    return f"Here is your recipe:\n\n{formatted_response}\n\nEnjoy your meal!"

def handle_prompt(prompt, df, knowledge_base, llm, conversation_chain):
    if prompt.lower().startswith("how to"):
        return generate_recipe(prompt, llm)
    else:
        response = process_query(df, knowledge_base, prompt, llm)
        conversation_chain.predict(input=prompt)
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
                    st.session_state.llm = get_llm()
                    st.session_state.conversation_chain = ConversationChain(
                        prompt=PromptTemplate(
                            input_variables=["history", "input"],
                            template="""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
                        ),
                        llm=st.session_state.llm,
                        verbose=True,
                        memory=ConversationBufferMemory(ai_prefix="AI Assistant")
                    )
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
                response = handle_prompt(
                    prompt,
                    st.session_state.df,
                    st.session_state.knowledge_base,
                    st.session_state.llm,
                    st.session_state.conversation_chain
                )
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Please upload a CSV first.")
