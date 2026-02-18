## 1. Introduction to Vector Stores

Vector stores are a critical component of the **RAG (Retrieval Augmented Generation)** architecture. They are specifically designed to solve the challenges of storing, managing, and searching through high-dimensional numerical representations of text known as embeddings.

---

## 2. Why Do We Need Vector Stores? (The Movie Case Study)

To understand the necessity of vector stores, consider building a movie recommendation system.

### The Flaw of Keyword Matching

Initial systems often use **Keyword Matching** (comparing directors, actors, or genres). However, this approach has significant flaws:

- **Logical Mismatch:** Two movies might share a director and actors (e.g., _My Name is Khan_ and _Kabhi Alvida Na Kehna_) but have completely different storylines, leading to poor recommendations.
- **Hidden Similarity:** Two movies might be very similar in theme (e.g., _Taare Zameen Par_ and _A Beautiful Mind_) but share no common keywords, causing the system to miss the connection.

### The Solution: Plot Similarity via Embeddings

A better approach is to compare the **semantic meaning** of movie plots.

1.  **Embeddings:** Use Deep Learning models to convert text plots into numerical **vectors** that represent their underlying meaning.
2.  **Vector Comparison:** These vectors are plotted in a multi-dimensional coordinate system.
3.  **Cosine Similarity:** Similarity is determined by the **angular distance** between vectors. A smaller distance indicates higher similarity.

---

## 3. Core Challenges Solved by Vector Stores

Building a large-scale semantic search system manually presents three major hurdles:

1.  **Massive Generation:** Generating embedding vectors for millions of documents.
2.  **Storage:** Traditional relational databases (MySQL, Oracle) are not designed to store vectors or perform mathematical similarity calculations efficiently.
3.  **Computationally Heavy Search:** Comparing a query vector against 10 million vectors one-by-one (linear search) is extremely slow and ruins user experience.

---

## 4. Key Features of Vector Stores

- **Storage & Metadata:** They store the vectors along with **metadata** (e.g., Movie ID, Title). They offer **In-memory** (for small, fast apps) or **On-disk** (for persistent, large-scale apps) storage.
- **Similarity Search:** The ability to instantly retrieve the most similar vectors to a given query.
- **Indexing (The "Smart Search"):** To avoid checking every single vector, they use techniques like **Clustering** or **ANN (Approximate Nearest Neighbors)**.
  - _Example:_ 1 million vectors are grouped into 10 clusters. A query is first compared to the 10 cluster centres (centroids). The system then only searches within the single most relevant cluster, reducing 1,000,000 comparisons to roughly 100,010.
- **CRUD Operations:** Easy methods to **C**reate (add), **R**etrieve, **U**pdate, and **D**elete vectors.

---

## 5. Vector Store vs. Vector Database

While often used interchangeably, there is a technical distinction:

- **Vector Store:** A lightweight system focused purely on storage and retrieval (e.g., Facebook’s **FAISS**).
- **Vector Database:** A full-fledged database system that adds enterprise features like **distributed architecture (scaling), backup/restore, ACID transactions, and authentication** (e.g., **Pinecone, Milvus, Weaviate**).

---

## 6. LangChain Implementation (Chroma DB Example)

LangChain provides a unified interface (wrappers) for all major vector stores. This allows you to switch from one (e.g., Chroma) to another (e.g., Pinecone) with minimal code changes.

### A. Initialization

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Define the embedding model and storage location
embedding_model = OpenAIEmbeddings()
persist_directory = "./my_chroma_db"

# Create/Initialize the vector store
vector_store = Chroma(
    collection_name="sample",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)
```

### B. Adding & Viewing Documents

```python
# Documents include page_content and metadata
docs = [Document(page_content="Virat Kohli is a batsman", metadata={"team": "RCB"})]

# Add to store (returns unique IDs for each doc)
ids = vector_store.add_documents(docs)

# Retrieve all stored data
data = vector_store.get(include=["embeddings", "documents", "metadatas"])
```

### C. Similarity Search

```python
# Simple Search
results = vector_store.similarity_search(query="Who is a bowler?", k=2)

# Search with Similarity Score (lower score = closer distance)
results_with_score = vector_store.similarity_search_with_score(query="Who is a bowler?", k=1)
```

### D. Metadata Filtering

```python
# Filter results based on metadata fields
mi_players = vector_store.similarity_search(
    query="",
    filter={"team": "Mumbai Indians"}
)
```

### E. Update & Delete

```python
# Update an existing document using its ID
vector_store.update_document(document_id=ids, document=updated_doc_object)

# Delete a document by ID
vector_store.delete(ids=[ids])
```

---
