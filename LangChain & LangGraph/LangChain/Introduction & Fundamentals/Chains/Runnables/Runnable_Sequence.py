# Runnable Sequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    RunnableSequence,
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()


def getBriefReport(topic: str):
    template = PromptTemplate(
        template="Give me a brief report on topic : {topic}", input_variables=["topic"]
    )
    formatted_prompt = template.format(topic=topic)
    return llm.invoke(formatted_prompt)


def getSummary(topic: str):
    template = PromptTemplate(
        template="Give me a brief summary of topic : {topic}", input_variables=["topic"]
    )
    res = llm.invoke(template.format(topic=topic))
    return res


chain1 = RunnableSequence(
    RunnableLambda(getBriefReport), RunnableLambda(getSummary), parser
)

chain2 = RunnableSequence(
    RunnableLambda(lambda x: "blockchain"),
    RunnableLambda(getBriefReport),
    RunnableLambda(getSummary),
    parser,
)


res1 = chain1.invoke({"topic": "blockchain"})
res2 = chain2.invoke({})

print(res1)
print()
print(res2)
