import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()
st.title("RAG For X-ray images using ChatGroq")
# Constants
DATA_FOLDER = "./data"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
VECTOR_STORE_PATH = "./vector_store"


def load_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            
            loader = PyPDFLoader(os.path.join(data_path,filename))
            documents += loader.load_and_split()

    return documents

# Test Constants
AGE = 58
GENDER = "Male"
DIAGNOSIS = "Cardiomelagy & Effusion"

if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3")
    
    if os.path.exists(VECTOR_STORE_PATH):
        st.session_state.vectors = FAISS.load_local(VECTOR_STORE_PATH, st.session_state.embeddings, allow_dangerous_deserialization=True)
        print("\nLoaded Vector Store from disk-----------\n")
    else:
        st.session_state.docs = load_documents(DATA_FOLDER)
        print("\nRead Documents ------------\n")

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Save vector store to disk
        st.session_state.vectors.save_local(VECTOR_STORE_PATH)
        print("\nSaved Vector Store to disk-----------\n")


llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


print("\nInitialised LLM -----------------\n")

prompt = ChatPromptTemplate.from_template(
"""
You are given a patient's data of Age, Gender and Diagnosis from X-ray images.
You are an experienced clinical assistant specializing in medical diagnostics and treatment recommendations. You can use the provided context for reference

<context>
{context}
<context>

Patient Data: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Enter Patient Data of Age, gender and Diagnosis")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Response time: ", (time.process_time()-start))
    st.write(response['answer'])
