# Runnable PassThrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()

chain = RunnableSequence(
    RunnableLambda(lambda x: 2),
    RunnableParallel(
        {
            "mul2": RunnableLambda(lambda x: x * 2),
            "mul3": RunnableLambda(lambda x: x * 3),
            "curr_num": RunnablePassthrough(),
        }
    ),
)

res = chain.invoke({})
print(res["curr_num"])  # 2
print(res["mul2"])  # 4
print(res["mul3"])  # 6
