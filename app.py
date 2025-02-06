import os
import json
import boto3
import streamlit as st

# LangChain and Bedrock Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_astradb import AstraDBVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Client Setup
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Set the environment variables (ensure they are set in your system or the runtime environment)
os.environ["ASTRA_DB_API_ENDPOINT"] =  "https://f22b7323-2a0b-4db4-8f74-1a8d41f2c1be-us-east1.apps.astra.datastax.com"
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = "AstraCS:mSnTIvWATclJenOWZHTzTzNq:27ab34debbb5d67aba5255b235a84f8a393c175cbb6f77ccb0de092642e30f46"

# Configure Astra DB Vector Store
def get_astra_vectorstore():
    # Correctly retrieve the token and endpoint from environment variables
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    
    if not token or not api_endpoint:
        raise ValueError("API endpoint or token is not set in environment variables.")

    return AstraDBVectorStore(
        collection_name="bolodocs_embeddings",  # Replace with your collection name
        embedding=bedrock_embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )

# PDF Processing
def process_pdf(uploaded_file):
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    
    os.remove(temp_path)
    return docs

# Store documents in Astra DB Vector Store
def store_in_astra(docs):
    vectorstore_astra = get_astra_vectorstore()
    vectorstore_astra.add_documents(docs)
    print("Documents stored in Astra DB vector store.")

# Load Astra DB Vector Store
def load_astra_vectorstore():
    return get_astra_vectorstore()

def get_llama2_llm():
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Generate Response
def get_response_llm(llm, vectorstore_astra, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_astra.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Main App


# Main app function
def main():
    # Set the page configuration for the Streamlit app
    st.set_page_config(
        page_title="BoloDocs",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for background color and text color styling
    st.markdown("""
        <style>
            body {
                background-color: #F5F5F5;  /* Light gray background */
                font-family: 'sans-serif';
            }
            .stApp {
                background-color: rgba(245, 245, 245, 0.9); /* Semi-transparent overlay */
            }
            h1 {
                text-align: center;
                color: #CD5C5C;  /* Text color set to CD5C5C (reddish) */
                font-size: 3rem;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            .description {
                text-align: center;
                color: #CD5C5C;  /* Text color for description */
                font-size: 1.2rem;
                margin-bottom: 20px;
            }
            .stButton>button {
                background-color: #6eb52f;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #5ca125;
            }

            /* Scrolling Quotes Styling */
            .quote-container-wrapper {
                display: flex;
                overflow: hidden;
                width: 100%;
                justify-content: center;
                align-items: center;
                padding: 20px 0;
                margin: 15px 0;
                position: relative;
            }

            .quote-container {
                display: inline-block;
                white-space: nowrap;
                padding: 20px 40px;
                margin: 0 30px;
                border: 2px solid #6eb52f;
                border-radius: 8px;
                background-color: rgba(240, 240, 245, 0.8);
                color: #6eb52f;
                font-size: 1.2rem;
                animation: scroll-left 18s linear infinite;
                animation-delay: 0s;
            }

            .quote-container2 {
                display: inline-block;
                white-space: nowrap;
                padding: 20px 40px;
                margin: 0 30px;
                border: 2px solid #e57373;
                border-radius: 8px;
                background-color: rgba(240, 240, 245, 0.8);
                color: #e57373;
                font-size: 1.2rem;
                animation: scroll-left 18s linear infinite;
                animation-delay: 6s;
            }

            .quote-container3 {
                display: inline-block;
                white-space: nowrap;
                padding: 20px 40px;
                margin: 0 30px;
                border: 2px solid #1E90FF;
                border-radius: 8px;
                background-color: rgba(240, 240, 245, 0.8);
                color: #1E90FF;
                font-size: 1.2rem;
                animation: scroll-left 18s linear infinite;
                animation-delay: 12s;
            }

            @keyframes scroll-left {
                0% {
                    transform: translateX(100%);
                }
                20% {
                    transform: translateX(0%);
                }
                60% {
                    transform: translateX(0%);
                }
                100% {
                    transform: translateX(-100%);
                }
            }

            /* Style the default file uploader component */
            .css-1lkd8ud {
                border: 2px dashed #6eb52f;  /* Green dashed border */
                border-radius: 8px;
                padding: 30px;
                background-color: rgba(245, 245, 245, 0.7);  /* Light gray background */
                color: #6eb52f;
                font-size: 1.2rem;
                text-align: center;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }

            .css-1lkd8ud:hover {
                background-color: rgba(245, 245, 245, 1);  /* Solid light gray on hover */
            }

            .css-1lkd8ud:active {
                background-color: rgba(245, 245, 245, 0.8);
                border-color: #4CAF50;
            }

        </style>
    """, unsafe_allow_html=True)

    # Main title and description
    st.markdown("<h1>Welcome to BoloDocs 💁</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="description">
            Ask questions about your PDF documents and get detailed answers with AI-powered processing.
        </div>
    """, unsafe_allow_html=True)

    # Scrolling quotes section
    st.markdown("""
        <div class="quote-container-wrapper">
            <div class="quote-container">
                "Transforming how you interact with your documents, one question at a time."
            </div>
            <div class="quote-container2">
                "Unleash the power of AI to extract knowledge from your PDFs."
            </div>
            <div class="quote-container3">
                "BoloDocs makes it easy to get instant answers from your PDF files."
            </div>
            <div class="quote-container">
                "AI-powered document processing for smarter work."
            </div>
            <div class="quote-container2">
                "Empowering you with the knowledge stored within your PDFs."
            </div>
            <div class="quote-container3">
                "Instant insights from your documents, simplified."
            </div>
            <div class="quote-container">
                "Access the data you need in a matter of seconds."
            </div>
            <div class="quote-container2">
                "AI that understands your documents for you."
            </div>
            <div class="quote-container3">
                "Bringing AI-powered document interaction to your fingertips."
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Upload and question input
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility="visible")
    user_question = st.text_input("Ask a Question from the PDF Files", placeholder="What would you like to ask?")

    # Sidebar: Manage vector store
    with st.sidebar:
        st.title("Update or Create Vector Store")
        st.markdown("Manage your knowledge base here.")
        if st.button("Update Vector Store"):
            if uploaded_file:
                with st.spinner("Processing and storing your PDF..."):
                    docs = process_pdf(uploaded_file)
                    store_in_astra(docs)
                    st.success(f"PDF '{uploaded_file.name}' successfully added to the knowledge base!")
            else:
                st.warning("Please upload a PDF file first to update the vector store.")

    # Query the PDF and generate a response
    if st.button("Show Me! 🤖"):
        if user_question:
            with st.spinner("Generating response..."):
                vectorstore_astra = load_astra_vectorstore()
                llm = get_llama2_llm()
                response = get_response_llm(llm, vectorstore_astra, user_question)
                st.write(response)
                st.success("Done!")
        else:
            st.warning("Please enter a question to get a response.")

if __name__ == "__main__":
    main()