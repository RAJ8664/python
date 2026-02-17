# Length based text-splitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

docs = TextLoader("temp.txt").load()

splitter = CharacterTextSplitter(chunk_size=15, chunk_overlap=0, separator="")

res = splitter.split_documents(docs)

for doc in res:
    print(doc.page_content)
