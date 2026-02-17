# Text Structure based text-splitter
"""
Order:
1. On the basis of paragraphs --> \n\n
2. On the basis of lines --> \b
3. On the basis of spaces --> ' '
4. On the basis of characters
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

docs = TextLoader("temp.txt").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=0)

res = splitter.split_documents(docs)

for doc in res:
    print(doc.page_content)
