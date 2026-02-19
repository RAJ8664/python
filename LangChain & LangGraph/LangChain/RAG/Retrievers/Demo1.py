from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatMessagePromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import Literal
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

wiki_retriever = WikipediaRetriever()

file = open("retrieved_data.txt", mode="w")

docs = wiki_retriever.invoke("Kathmandu Nepal")

for doc in docs:
    file.write(doc.page_content)

file = open("retrieved_data.txt", mode="r")
for line in file:
    print(line)

file.close()
