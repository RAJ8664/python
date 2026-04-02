from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatMessagePromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.retrievers import WikipediaRetriever
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

embedding_model = OpenAIEmbeddings()

docs = TextLoader("retrieved_data.txt").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

vector_store = Chroma(
    persist_directory="DB",
    collection_name="my_data",
    embedding_function=embedding_model,
)

docs = text_splitter.split_documents(
    docs
)  # Perform splitting smartly otherwise the result will not be very good.

vector_store.add_documents(docs)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.25},  # Read the official docs for more
)

query = "explain the history of the Nepal"

res = retriever.invoke(query)

print(res)


# Note : There are multiple retrievers each having their own algorithm (Refer to official langchain docs).
