# Expole of embeddings in langchain using documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
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

query = "Who is virat kohli?"

# Find the embedding vector for query and document.
query_embedding = embeddings.embed_query(query)
doc_embeddings = embeddings.embed_documents(documents)

# Find closest document
best_idx = min(
    range(len(doc_embeddings)),
    key=lambda i: np.linalg.norm(
        np.array(doc_embeddings[i]) - np.array(query_embedding)
    ),
)

print("Most relevant document:")
print(documents[best_idx])
