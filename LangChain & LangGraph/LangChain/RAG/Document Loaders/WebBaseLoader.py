# WebBaseLoader
"""
1. WebBase Loader is a document loader in LangChain used to load and extract text content from web pages(URLs).
2. It uses BeautifulSoup under the hood to parser HTML and extract visible text.
3. Used mostly for blogs, news articles, or public websites where the content is primarily text-based and static
4. Does not handle JavaScript-heavy pages well(user SeleiumURLLoader for that).
5. Loads only static content(what's in the HTML, not what loads after the page renders).
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")
parser = StrOutputParser()

url = "https://raj8664.github.io/Prep/branch/cs/SEM3/CS201/PYQ/Mid-Semester/"
loader = WebBaseLoader(url)  # List of URLs can also be passed here.
docs = loader.load()

prompt = PromptTemplate(
    template="Explain the following text: \n {qpaper}",
    input_variables=["qpaper"],
)

chain = prompt | llm | parser

res = chain.invoke({"qpaper": docs[0].page_content})

print(res)
