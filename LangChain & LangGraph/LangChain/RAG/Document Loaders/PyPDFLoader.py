# PyPDFLoader
"""
1. PyPDFLoader is a document loader in LangChain used to load content from PDF files and convert each page into a Document object.
2. It uses the PyPDF library under the hood -> not great with scanned PDFs or complex layouts.
3. Other Better PDFs loaders documentation can be found here --> https://docs.langchain.com/oss/python/integrations/document_loaders
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

loader = PyPDFLoader("/home/rkroy/Downloads/5th sem payment 2.pdf")

docs = loader.load()


def getAllLinks(docs):
    arr = []
    res = []
    for doc in docs:
        temp = doc.page_content.split()
        for word in temp:
            if word.__contains__("http"):
                res.append(word)
    return res


print(len(docs))
print()

allLinks = getAllLinks(docs)

print(allLinks)
