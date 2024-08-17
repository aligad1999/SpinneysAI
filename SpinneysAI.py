import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.llms import HuggingFaceHub
import fuzzywuzzy
from fuzzywuzzy import process
import re

def process_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    # Ensure all product titles are strings and handle NaNs
    df['product_title'] = df['Product Name'].fillna('').astype(str).str.lower().str.strip()

    # Create embeddings and knowledge base
    texts = list(map(lambda x: x.replace("\n", " "), df['product_title'].tolist()))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts, embeddings)
    
    return df, knowledge_base

def get_llm():
    # Provide your API key directly here
    api_key = "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"
    return HuggingFaceHub(
        repo_id="gpt2-xl",
        model_kwargs={"temperature": 0.5, "max_length": 1024},
        huggingfacehub_api_token=api_key
    )

def process_query(df, knowledge_base, query, llm):
    # Retrieve multiple similar matches
    docs = knowledge_base.similarity_search(query, k=10)
    results = []
    seen_titles = set()
    
    # Convert document contents to lower case for matching
    doc_titles = [doc.page_content.lower() for doc in docs]
    
    # Fuzzy matching to find best matches
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
    # Format the prompt to guide the LLM in providing a concise, structured recipe
    formatted_prompt = (
        "Generate a warm, friendly, and concise recipe for the following request. "
        "Include the ingredients, equipment, and step-by-step instructions, as well as any useful tips. "
        "Please keep the response free from unnecessary details or repeated information.\n\n"
        "{}"
    ).format(prompt)
    
    # Call the LLM with the formatted prompt
    raw_response = llm(formatted_prompt)
    
    # Remove the instructional lines using regex
    regex_pattern = r"(Generate a warm, friendly, and concise recipe for the following request\.|Include the ingredients, equipment, and step-by-step instructions, as well as any useful tips\.|Please keep the response free from unnecessary details or repeated information\.)"
    cleaned_response = re.sub(regex_pattern, '', raw_response).strip()
    
    # Ensure the prompt itself isn't visible in the output
    cleaned_response = cleaned_response.replace(prompt, "").strip()

    # Check if the response seems incomplete or if the necessary sections are missing
    sections = ["Ingredients:", "Equipment:", "Instructions:", "Tips:"]
    missing_sections = [section for section in sections if section not in cleaned_response]

    if len(cleaned_response) < 200 or missing_sections:
        additional_prompt = (
            "It seems some parts of the recipe might be missing or incomplete. Could you please provide the missing details or "
            "expand on the recipe to include the following sections: {}?"
        ).format(", ".join(missing_sections))
        
        additional_response = llm(additional_prompt)
        cleaned_response += "\n" + additional_response

    # Final formatting to ensure no repetitions and correct output structure
    lines = cleaned_response.splitlines()
    seen_lines = set()
    final_response = []
    for line in lines:
        if line.strip() and line not in seen_lines:
            final_response.append(line)
            seen_lines.add(line)
    
    final_output = "\n".join(final_response).strip()

    # Ensure the final response is friendly and well-formatted
    if final_output and final_output[-1] != "!":
        final_output += "!"

    final_output = f"Here's a lovely recipe just for you:\n\n{final_output}\n\nEnjoy your delicious creation!"

    return final_output





def handle_prompt(prompt, df, knowledge_base, llm):
    if prompt.lower().startswith("how to"):
        return generate_recipe(prompt, llm)
    else:
        return process_query(df, knowledge_base, prompt,llm)


if __name__ == '__main__':
    st.set_page_config(page_title="Chat with Spinneys AI", page_icon=":shark:", layout="wide")
    st.title("Chat with Spinneys AI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload CSV")

        # Check if the CSV data is already in session_state
        if 'df' not in st.session_state or 'knowledge_base' not in st.session_state:
            csv_file = st.file_uploader("Upload your CSV File", type="csv")
            
            if csv_file is not None:
                with st.spinner("Processing..."):
                    df, knowledge_base = process_csv(csv_file)
                    st.session_state.df = df
                    st.session_state.knowledge_base = knowledge_base
                    st.session_state.llm = get_llm()
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
                response = handle_prompt(prompt, st.session_state.df, st.session_state.knowledge_base, st.session_state.llm)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Please upload a CSV first.")
