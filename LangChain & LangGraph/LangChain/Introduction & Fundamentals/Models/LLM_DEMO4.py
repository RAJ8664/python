# Example of embeddings in langchain using documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

documents = [
    "Virat Kohli is a renowned Indian cricketer known for his aggressive batting style and leadership.",
    "Sachin Tendulkar, often called the 'God of Cricket', is a legendary Indian batsman.",
    "Rohit Sharma is an Indian cricketer famous for his elegant batting.",
    "Jasprit Bumrah is a leading Indian fast bowler known for his unique action.",
    "MS Dhoni is a former Indian captain celebrated for his calm demeanor.",
]
query = "who is the cricket known for best indian captain and who always looks cool and calm"

# Find the embedding vector for query and document.
query_embedding = embeddings.embed_query(query)
doc_embeddings = embeddings.embed_documents(documents)

# We will try to find the similarity scores of query_embedding with each query_embedding[i]
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)
# Similarity_scores will be a vector of size(1, len(documents)) each element of which is the cosine similarity score of query_embedding with doc_embeddings[i]
print(similarity_scores)

similarity_scores = similarity_scores[0]  # convert [[.., .., ..]] to [.., .., ..]

# Index with highest value will be the most relevant document
best_idx = 0
best_score = similarity_scores[0]

for i in range(0, len(similarity_scores)):
    if similarity_scores[i] > best_score:
        best_score = similarity_scores[i]
        best_idx = i

print("Most relevant document:")
print(documents[best_idx])
