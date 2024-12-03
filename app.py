import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import json

load_dotenv()

# Load API keys
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings and language model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create vector embeddings during app initialization
@st.cache_resource
def create_vector_embedding():
    loader = PyPDFDirectoryLoader("research_papers")  # Data Ingestion step
    docs = loader.load()  # Document Loading
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

st.title("RAG Document Q&A With Groq And Llama3")

# Initialize vector database
st.session_state.vectors = create_vector_embedding()

# Query input
user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    import time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")

    # Display the answer
    st.write(response['answer'])

    # With a Streamlit expander for document similarity search
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

# Add an API endpoint for React integration
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")
    if user_query:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_query})
        return jsonify({"answer": response['answer']})
    else:
        return jsonify({"error": "No query provided"}), 400
