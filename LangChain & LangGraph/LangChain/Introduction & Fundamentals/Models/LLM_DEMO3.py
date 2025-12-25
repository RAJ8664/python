# This script demonstrates how to use Google Generative AI Embeddings with LangChain
# to generate an embedding vector for a given query string.
#
# Steps:
# 1. Load environment variables (such as API keys) from a .env file.
# 2. Initialize the GoogleGenerativeAIEmbeddings model with the desired embedding model.
# 3. Generate an embedding for a sample query.
# 4. Print the resulting embedding vector.

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)  # Initialize embedding model

res = llm.embed_query(
    "what is the capital of Nepal?"
)  # Generate embedding for the query

print(str(res))  # Print the embedding vector
