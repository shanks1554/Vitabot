import os
import re
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Streamlit app configuration
st.set_page_config(page_title="VitaBot", page_icon="🕊️", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    .stChatMessage {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        display: inline-block; /* Only as wide as the text */
        max-width: 60%; /* Ensures it doesn't become too wide */
        text-align: left;
        color: white;
        background-color: #1a73e8;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        word-wrap: break-word; /* Prevents overflow */
    }
    .assistant-message {
        display: inline-block;
        max-width: 60%;
        background-color: #444654;
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"},
    )

# Custom chatbot prompt
CUSTOM_PROMPT_TEMPLATE = """
You are an AI healthcare assistant specializing in medicinal plants, herbal remedies, and spiritual healing.  
Your goal is to provide **clear, empathetic, and accurate** information while ensuring user safety.  

- If the user greets you (e.g., "Hello," "Hi"), respond warmly with **"Hello! How can I assist you today?"**  
- If the user describes symptoms or asks for a cure, **never diagnose or assume a condition.** Instead, respond with:  
  **"I'm not a doctor, but I can provide general wellness tips. For a proper diagnosis and treatment, please consult a qualified healthcare professional."**  
- If the user asks about herbal remedies, describe their **traditional uses** but always include a disclaimer:  
  **"This is not a substitute for professional medical advice. Please consult a doctor before using herbal treatments."**  
- Maintain a **friendly, supportive, and professional** tone in all responses.  

Context:{context}  
Question:{question}  

Answer:
"""

# Main application logic
def main():
    st.title("💊 Ask VitaBot")
    st.write("**Your AI assistant for medical assistance ⚕️**")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

    # User input
    prompt = st.chat_input("Message VitaBot")

    if prompt:
        # Display user message
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("⚠️ Failed to load the vector store.")
                return

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )

            # Get response from model
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            # Remove unnecessary question parts
            result = re.sub(r"Question:.*?\n", "", result, flags=re.DOTALL).strip()

            # Display assistant message
            st.markdown(f"<div class='assistant-message'>{result}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
