# Example: Using LangChain and Google Generative AI Embeddings to find the most relevant document to a query

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file (for API keys, etc.)
load_dotenv()

# Initialize the Google Generative AI Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# List of documents to compare against the query
documents = [
    "Virat Kohli is a renowned Indian cricketer known for his aggressive batting style and leadership.",
    "Sachin Tendulkar, often called the 'God of Cricket', is a legendary Indian batsman.",
    "Rohit Sharma is an Indian cricketer famous for his elegant batting.",
    "Jasprit Bumrah is a leading Indian fast bowler known for his unique action.",
    "MS Dhoni is a former Indian captain celebrated for his calm demeanor.",
]

# The query to search for the most relevant document
query = "who is the cricket known for best indian captain and who always looks cool and calm ?"

# Get the embedding vector for the query
query_embedding = embeddings.embed_query(query)

# Get the embedding vectors for all documents
doc_embeddings = embeddings.embed_documents(documents)

# Compute cosine similarity between the query embedding and each document embedding
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)

# similarity_scores is a 2D array; flatten it to 1D
similarity_scores = similarity_scores[0]

# Find the index of the document with the highest similarity score
best_idx = 0
best_score = similarity_scores[0]
for i in range(0, len(similarity_scores)):
    if similarity_scores[i] > best_score:
        best_idx = i
        best_score = similarity_scores[i]

print("Most relevant document:")
print(documents[best_idx])
