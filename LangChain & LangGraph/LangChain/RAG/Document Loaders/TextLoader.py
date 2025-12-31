# TextLoader
"""
1. TextLoader is a simple and commonly used document loader in LangChain that reads plain text(.txt) files and converts them into LangChain Document objects.
2. Ideal for loading chat logs, scraped text, transcripts, code snippets, or any plain text data into a LangChain pipeline.
3. works only with .txt files.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

loader = TextLoader(
    "/home/rkroy/Desktop/code/python/requirements.txt", encoding="utf-8"
)
docs = (
    loader.load()
)  # docs = list( Document1(), Document1(), Document1()) --> Document = Document(page_content, metadata)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="I want you to explain every python libraries mentioned in this given text one by one. \n {text}",
    input_variables=["text"],
)

chain = prompt | llm | parser  # RunnableSequence(prmopt, llm, parser)

res = chain.invoke({"text": docs[0].page_content})

print(res)


"""
output : 

Here's a detailed explanation of each Python library mentioned in the text, grouped by category:

---

### **LangChain Ecosystem**
1. **langchain**  
   The core library for building LLM-powered applications. Provides abstractions for chains, agents, tools, and memory management to connect LLMs with external data/system interactions.  
   *Use case: Building chatbots with retrieval-augmented generation (RAG).*

2. **langchain-core**  
   Minimal base interfaces and core functionality for LangChain components. Serves as the foundation for LangChain's higher-level modules.  
   *Use case: Creating custom LangChain components.*

3. **langchain-community**  
   Third-party integrations (tools, vector stores, LLMs) for LangChain from the community. Simplifies adapting LangChain to external services.  
   *Use case: Integrating LangChain with niche APIs/databases.*

---

### **OpenAI Integration**
4. **langchain-openai**  
   Official LangChain integration suite for OpenAI models. Includes LLM wrappers, embedding tools, and chat interfaces for GPT models.  
   *Use case: Using GPT-4-turbo in a LangChain agent.*

5. **openai**  
   OpenAI's official SDK to interact with their APIs (ChatGPT, DALL·E, Whisper, embeddings).  
   *Use case: Direct API calls to OpenAI models.*

---

### **Google Gemini Integration**
6. **langchain-google-genai**  
   LangChain integration for Google's Gemini models. Provides LLMs, embedding support, and chat tools.  
   *Use case: Building RAG apps with Gemini Pro.*

7. **google-generativeai**  
   Google's official SDK for Gemini API access. Offers model inference and safety settings configuration.  
   *Use case: Generating text/images with Gemini Pro via API.*

---

### **HuggingFace Integration**
8. **langchain-huggingface**  
   Integration tools for using Hugging Face models (LLMs, embeddings) within LangChain pipelines.  
   *Use case: Running local LLMs (e.g., Llama 3) via LangChain.*

9. **transformers**  
   Hugging Face's NLP library for state-of-the-art transformer models (BERT, GPT-like). Includes APIs for inference/training.  
   *Use case: Fine-tuning a BERT model for text classification.*

10. **huggingface_hub**  
    Python client for interacting with Hugging Face Hub (download models/datasets, push trained models).  
    *Use case: Downloading pre-trained models like Mistral-7B.*

---

### **Environment Management**
11. **python-dotenv**  
    Loads environment variables from `.env` files into Python applications. Keeps secrets/tokens secure and out of code.  
    *Use case: Storing API keys safely outside code.*

---

### **Machine Learning Utilities**
12. **numpy**  
    Foundational library for scientific computing. Provides n-dimensional arrays and math operations essential for ML/LLM workflows.  
    *Use case: Preprocessing text embeddings.*

13. **scikit-learn**  
    Machine learning toolkit for classification/regression/clustering. Includes text vectorizers like TF-IDF.  
    *Use case: Building classifiers for text data.*

---

### **Web UI Framework**
14. **streamlit**  
    Framework for rapid ML/data app development. Creates interactive UIs with minimal Python-only code.  
    *Use case: Deploying a LangChain RAG app with a web interface.*

---

### **Visualization**
15. **grandalf**  
    Graph and drawing framework. Used in LangChain for visualizing chain workflows (`render_graph()` method).  
    *Use case: Visualizing sequence of steps in a LangChain pipeline.*

---

### **Key Relationships**
- **LangChain** leverages **langchain-core**, **langchain-community**, and model-specific libraries (**langchain-openai**, **langchain-google-genai**, etc.) to build applications.  
- **Streamlit** enables web UIs for LangChain apps.  
- **Transformers**/**HuggingFace**/**Google-GenerativeAI** provide model access.  
- **python-dotenv** manages secrets for APIs.  
- **numpy**/**scikit-learn** support preprocessing/prediction tasks.  
- **grandalf** aids in debugging LangChain flows.

"""
