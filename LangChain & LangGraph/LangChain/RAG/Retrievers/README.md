## 1. Introduction to Retrievers

**Retrievers** are a fundamental component of **RAG (Retrieval Augmented Generation)** applications. A retriever is a LangChain component designed to fetch relevant documents from a data source in response to a user's query. While a data source stores all the information, the retriever acts as a search engine that scans the data, identifies the most relevant items, and returns them as a list of **Document objects**.

### Key Characteristics:

- **Input/Output:** A retriever takes a **string (query)** as input and returns a **list of Document objects** as output.
- **Runnables:** Every retriever in LangChain is a **Runnable**. This means they can be integrated into **Chains** using the pipe operator (`|`) and possess an `invoke()` method.
- **Flexibility:** Because they are runnables, they can be easily plugged into existing RAG pipelines to enhance system performance.

---

## 2. Categorization of Retrievers

Retrievers are categorized based on two main criteria:

### A. Based on Data Source

Different retrievers are designed to work with specific data sources.

- **Wikipedia Retriever:** Fetches content directly from the Wikipedia API.
- **Vector Store Retriever:** Searches for documents within a vector database (like Chroma or FAISS).
- **Arxiv Retriever:** Scans research papers on the Arxiv website to find relevant information.

### B. Based on Search Strategy (Mechanism)

Retrievers can also be distinguished by the specific logic they use to find relevant documents.

- **MMR (Maximum Marginal Relevance):** Focuses on diversity to avoid redundant results.
- **Multi-Query Retriever:** Generates multiple versions of a query to handle ambiguity.
- **Contextual Compression:** Trims documents to keep only the relevant parts.

---

## 3. Specific Retrievers and Implementation

### 1. Wikipedia Retriever

This retriever hits the Wikipedia API to fetch articles related to a query. It primarily uses **keyword matching** rather than semantic search to decide relevance.

**Example Code:**

```python
from langchain_community.retrievers import WikipediaRetriever

# Create the retriever object
# 'top_k_results' defines how many articles to return
retriever = WikipediaRetriever(top_k_results=2)

# Using the invoke method (proving it is a Runnable)
docs = retriever.invoke("The geopolitical history of India and Pakistan")

# Accessing the content
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

_Note: A retriever differs from a Document Loader because it doesn't load everything; it performs an intelligent search to find specific relevant documents based on a query._

### 2. Vector Store Retriever

This is the most common retriever type, used to fetch documents from a vector store based on **semantic similarity** using embeddings.

**Example Code:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Setup Vector Store (e.g., Chroma)
vector_store = Chroma.from_documents(documents, OpenAIEmbeddings())

# 2. Create the Retriever object
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. Fetch documents
query = "What is Chroma used for?"
results = retriever.invoke(query)
```

**Why use a Retriever over `similarity_search`?** While a vector store can perform a similarity search directly, a Retriever object allows you to implement **advanced search strategies** and integrate seamlessly into LangChain **Chains**.

### 3. Maximum Marginal Relevance (MMR)

Standard similarity searches often return **redundant** results (e.g., three documents saying the exact same thing). MMR solves this by selecting documents that are both **relevant** to the query and **diverse** from each other.

**Mechanism:** It picks the most relevant document first, then picks the next document that is relevant but most _dissimilar_ to the first one.

- **Lambda Parameter:** Varies from 0 to 1.
  - **1:** Behaves like a standard similarity search.
  - **0:** Provides maximum diversity in results.

**Example Code:**

```python
# Using a vector store (e.g., FAISS) to create an MMR retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'lambda_mult': 0.5}
)
```

### 4. Multi-Query Retriever

Users often provide **ambiguous or broad queries** (e.g., "How can I stay healthy?") which are difficult to match against specific documents.

**Mechanism:**

1.  The query is sent to an **LLM**.
2.  The LLM generates **multiple variations** of the query to cover different angles (e.g., diet, exercise, stress).
3.  The system runs all queries through a base retriever, merges the results, and removes duplicates.

**Example Code:**

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Initialize with an LLM and a base retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)
results = retriever.invoke("How to improve energy levels and maintain balance")
```

### 5. Contextual Compression Retriever

Sometimes a relevant document is very long and contains "mixed information" (e.g., a paragraph about the Grand Canyon followed by a sentence about photosynthesis). Returning the whole document wastes space and can confuse the LLM.

**Mechanism:**

1.  A **Base Retriever** fetches relevant documents.
2.  A **Compressor (usually an LLM)** analyzes these documents alongside the user query.
3.  The compressor **trims or extracts** only the sentences that are relevant to the query, discarding the rest.

**Example Code:**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 1. Define base retriever and compressor (LLM)
base_retriever = vector_store.as_retriever()
compressor = LLMChainExtractor.from_llm(llm)

# 2. Create the compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 3. Result will be short, extracted sentences
results = compression_retriever.invoke("What is photosynthesis?")
```

---

## 4. Summary: The Role of Retrievers in Advanced RAG

The primary reason so many retrievers exist is to **improve RAG system performance**. If a basic RAG system provides poor results, developers can swap standard retrievers for advanced ones (like MMR or Multi-Query) to rebuild and optimize the system. Understanding these retrievers is the key to moving from basic to **Advanced RAG** applications.
