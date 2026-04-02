from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
from langchain_community.tools import ShellTool, DuckDuckGoSearchRun

load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
embedding_model = OpenAIEmbeddings()

shell_tool = ShellTool()
search_tool = DuckDuckGoSearchRun()

files = shell_tool.invoke("ls -lah")
working_dir = shell_tool.invoke("pwd")
search = search_tool.invoke("Albert Einstein")

print(files)
print(working_dir)
print(search)
