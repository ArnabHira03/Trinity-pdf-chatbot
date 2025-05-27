import streamlit as st
import os
import time
import re
import tempfile
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Trinity v0.1")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader("Drop a PDF here", type="pdf", accept_multiple_files=True)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load_and_split()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return chunks

if uploaded_files:
    all_chunks = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, uploaded_files)
        for chunks in results:
            all_chunks.extend(chunks)


    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    db = FAISS.from_documents(all_chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")
    

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    st.session_state.qa_chain = qa_chain
else:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")
    st.session_state.llm = llm

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])



user_prompt = st.chat_input("Ask something...")

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "text": user_prompt})

    tic = time.perf_counter()

    if uploaded_files and "qa_chain" in st.session_state:
        raw_output = st.session_state.qa_chain.run(user_prompt)
    else:
        raw_output = st.session_state.llm.invoke(user_prompt).content

    answer = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    toc = time.perf_counter()

    with st.chat_message("assistant"):
        st.markdown(answer + f"\n\n*⏱ {toc - tic:.2f}s*")
    st.session_state.messages.append({"role": "assistant", "text": answer})
