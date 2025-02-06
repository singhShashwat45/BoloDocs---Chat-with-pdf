import cassio
import os
from getpass import getpass

# Enter your settings for Astra DB and OpenAI:
os.environ["ASTRA_DB_API_ENDPOINT"] ="https://f22b7323-2a0b-4db4-8f74-1a8d41f2c1be-us-east1.apps.astra.datastax.com"
os.environ["ASTRA_DB_APPLICATION_TOKEN"] ="AAstraCS:bXzKgQAeEoDolSiXCnAaCfHe:e75c989c9408080e5c6c88ee2d52e045c2534b7706154c3d61e0da4dfa78e1cd"
os.environ["OPENAI_API_KEY"] = "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"

from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import OpenAIEmbeddings
import os

# Configure your embedding model and vector store
embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    collection_name="bolodocs",
    embedding=embedding,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
print("Astra vector store configured")