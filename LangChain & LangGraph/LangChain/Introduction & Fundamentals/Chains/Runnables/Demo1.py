from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")


def addTwo(x: int):
    return x + 2


def square(x: int):
    return x**2


chain1 = RunnableLambda(addTwo) | RunnableLambda(square)

chain2 = RunnableLambda(addTwo) | {
    "mul1": RunnableLambda(lambda x: x * 2),
    "mul2": RunnableLambda(lambda x: x * 3),
    "mul3": RunnableLambda(lambda x: x * 4),
}


print(chain1.invoke(1))  # 9
print(chain2.invoke(1))  # {'mul1': 6, 'mul2': 9, 'mul3': 12}
