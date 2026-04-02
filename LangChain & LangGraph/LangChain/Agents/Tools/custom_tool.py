from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from dotenv import load_dotenv
from langchain_community.tools import BaseTool, Tool, ShellTool, tool

load_dotenv()

embedding_model = OpenAIEmbeddings()


@tool
def add_two(a: int, b: int) -> int:
    """
    Used to add two numbers a and b
    """

    return a + b


print(add_two.name)

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

query = "add two number 3 and 4"
res = chat_model.invoke(query)

print(res)
