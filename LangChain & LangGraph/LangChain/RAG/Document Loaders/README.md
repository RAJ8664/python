# Document Loaders in LangChain: Comprehensive Notes

## 1. Introduction to RAG (Retrieval Augmented Generation)

While Large Language Models (LLMs) are powerful, they have significant limitations, such as being trained on past data (knowledge cutoff) and lacking access to personal or company-specific private data. **Retrieval Augmented Generation (RAG)** is a technique that addresses these issues by combining information retrieval from an external knowledge base with language generation from an LLM.

In a RAG-based application, the model retrieves relevant documents from a knowledge base and uses them as context to generate accurate, grounded, and up-to-date responses. The four core components of a RAG architecture are:

1.  **Document Loaders**
2.  **Text Splitters**
3.  **Vector Databases**
4.  **Retrievers**

---

## 2. Understanding Document Loaders

**Document Loaders** are utilities in LangChain designed to load data from various sources (PDFs, text files, databases, cloud providers) into a **standardised format** called a **Document object**.

### The Document Object Structure

Every data source loaded via a document loader is converted into a standard Document object containing two main parts:

- **`page_content`**: The actual textual content of the data.
- **`metadata`**: Information about the data, such as the source, author, creation date, or page number.

All document loaders are housed within the `langchain_community.document_loaders` package.

---

## 3. Core Document Loaders & Code Examples

### A. Text Loader

The simplest loader used for `.txt` files, logs, or code snippets.

```python
from langchain_community.document_loaders import TextLoader

# Initialise loader with file path and optional encoding
loader = TextLoader("cricket.txt", encoding="utf-8")

# Load returns a List of Document objects
docs = loader.load()

# Accessing content
print(docs.page_content)
print(docs.metadata)
```

_Note: Even for a single file, loaders return a **List** of documents_.

### B. PyPDF Loader

This loader reads PDF files and processes them on a **page-by-page** basis. If a PDF has 23 pages, it will generate a list of 23 Document objects, each with its own page number in the metadata.

```python
# Requires: pip install pypdf
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("curriculum.pdf")
docs = loader.load()

print(len(docs)) # Returns number of pages
print(docs.metadata) # Includes 'page' number and 'source'
```

_Limitations: It is best for textual PDFs; it may struggle with scanned images or complex layouts_.

### C. Directory Loader

Used to load multiple files from a folder simultaneously using a search pattern (glob).

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all PDFs within the 'books' folder
loader = DirectoryLoader(
    "books/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(f"Total pages loaded: {len(docs)}")
```

Patterns like `**/*.txt` can be used to load all text files from subdirectories.

### D. WebBase Loader

Extracts text content from HTML web pages using `Requests` and `BeautifulSoup`.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.example.com/product")
docs = loader.load()

# Extracts text while ignoring HTML tags
print(docs.page_content)
```

_Best for static websites (blogs, news); less effective for JavaScript-heavy sites_.

### E. CSV Loader

Converts a CSV file into Document objects, where **each row** becomes a separate document.

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
docs = loader.load()

# Each document represents one row with column headers as keys
print(docs.page_content)
```

---

## 4. Eager vs. Lazy Loading

Loading thousands of documents into RAM simultaneously can crash a system. LangChain provides two methods:

| Method            | Type          | Description                                                                                      |
| :---------------- | :------------ | :----------------------------------------------------------------------------------------------- |
| **`load()`**      | **Eager**     | Loads everything into memory at once. Returns a **List**. Best for small datasets.               |
| **`lazy_load()`** | **On-demand** | Fetches one document at a time. Returns a **Generator**. Best for large-scale data or streaming. |

---

## 5. Specialized Loaders & Customization

- **Specialized PDFs**: Use `PDFPlumberLoader` for tables, `UnstructuredPDFLoader` for scanned images, or `PyMuPDF` for complex layouts.
- **Cloud/Platforms**: Loaders exist for **S3, Google Drive, Azure, GitHub, Slack**, and **YouTube transcripts**.
- **Custom Loaders**: If a loader doesn't exist, you can create one by inheriting from the `BaseLoader` class and defining your own `load` and `lazy_load` logic.

---

**Analogy for Understanding:**
Think of **Document Loaders** like a **universal translator** for a library. Whether the information arrives as a handwritten scroll (Text file), a bound book (PDF), a spreadsheet (CSV), or a radio broadcast (Web page), the loader translates them all into the same "Standard Language" (Document Object) so the librarian (the LLM) can read and organize them easily.
