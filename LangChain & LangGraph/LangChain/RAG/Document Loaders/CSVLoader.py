# CSVLoader
"""
CSV Loader is a document loader in LangChain used to load CSV files into LangChain Document object one per row by default.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    CSVLoader,
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

loader = CSVLoader("/home/rkroy/Desktop/code/python/test/data/csv/sample.csv")

docs = loader.load()

print(len(docs))
print(docs)
