## 1. Introduction to the Models Component

The **Models component** is a core part of the LangChain framework designed to facilitate interactions with various AI models. It acts as a **common interface**, allowing developers to connect to different AI models from various companies without writing unique code for each.

### Types of Models in LangChain

There are two primary types of models supported:

- **Language Models:** These take text as input and return text as output (e.g., answering "What is the capital of India?" with "New Delhi").
- **Embedding Models:** These take text as input but return a **series of numbers (vectors)** known as embeddings. These represent the contextual meaning of the text and are used for **semantic search** and **RAG-based (Retrieval Augmented Generation) applications**.

---

## 2. LLMs vs. Chat Models

Language models are further divided into two categories:

| Feature            | LLMs (Large Language Models)                                | Chat Models                                                    |
| :----------------- | :---------------------------------------------------------- | :------------------------------------------------------------- |
| **Input/Output**   | String in, String out.                                      | Sequence of messages in, message out.                          |
| **Purpose**        | General-purpose NLP tasks (summarization, code generation). | Specialized for conversational tasks (chatbots, agents).       |
| **Training**       | General text (books, articles).                             | Fine-tuned on chat datasets (Reddit, WhatsApp chats).          |
| **Memory**         | No inherent memory of previous conversations.               | Supports conversation history.                                 |
| **Role Awareness** | No roles.                                                   | Understands roles: **System**, **User**, and **AI/Assistant**. |

**Note:** LLMs are becoming legacy in LangChain; the industry and the library now recommend using **Chat Models** for new projects.

---

## 3. Environment Setup

To follow the practical demonstration, a proper environment is required:

1.  **Create a Folder:** e.g., `langchain_models` or anything you like.
2.  **Virtual Environment:**
    - Create: `python -m venv venv`.
    - Activate: `venv\Scripts\activate` (Windows).
    - Activate: `source venv/bin/activate` (Linux).
3.  **Installation:** Create a `requirements.txt` file and install libraries using `pip install -r requirements.txt`.(Mention Required Libraries in the requirements.txt file)
4.  **Verification:** Import `langchain` and print its version to ensure a successful install.

---

## 4. Working with Language Models

### A. Closed Source Models (Paid APIs)

Using these requires an **API Key** stored in a `.env` file for security.

#### OpenAI

- **LLM Example:** Uses `langchain-openai` and the `OpenAI` class with models like `gpt-3.5-turbo-instruct`.
- **Chat Model Example:** Uses `ChatOpenAI` with `gpt-4`.
- **The `invoke()` method:** This is the standard method in LangChain to communicate with models.
- **Parameters:**
    - **Temperature:** Controls creativity (0 is deterministic/factual, 1.5+ is creative/random).
    - **Max Tokens:** Restricts the length of the output to manage costs.

#### Anthropic (Claude)

- **Library:** `langchain-anthropic`.
- **Class:** `ChatAnthropic`.
- **Model Used:** `claude-3-5-sonnet` (or latest versions).

#### Google (Gemini)

- **Library:** `langchain-google-genai`.
- **Class:** `ChatGoogleGenerativeAI`.
- **Model Used:** `gemini-1.5-pro`.

### B. Open Source Models

Open-source models offer **full control**, **data privacy** (local processing), and **no API costs**, but require **powerful hardware (GPUs)** and can be complex to set up.

#### Hugging Face Integration

- **Inference API:** Uses Hugging Face's servers via an API key. Uses `HuggingFaceEndpoint`.
- **Local Execution:** Downloads the model weights to your machine using `HuggingFacePipeline`.
- **Demonstration Model:** `TinyLlama` (approx. 1.1 billion parameters).

---

## 5. Working with Embedding Models

Embedding models convert text into vectors for mathematical comparison.

- **OpenAI Embeddings:** Uses `OpenAIEmbeddings` class. You can specify the `dimensions` (e.g., 32, 300, or the default 1536/3072).
- **Methods:**
    - `embed_query`: For a single sentence.
    - `embed_documents`: For a list of multiple texts.
- **Hugging Face (Local):** Uses `HuggingFaceEmbeddings` with models like `all-MiniLM-L6-v2`. This is useful for free, local vector generation.

---

## 6. Practical Application:

- See the Code implementations in the same repository for better understanding.
