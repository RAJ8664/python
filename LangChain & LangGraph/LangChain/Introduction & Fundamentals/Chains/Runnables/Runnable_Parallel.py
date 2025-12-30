# Runnable Parallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda
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


def getShortSummary(topic: str):
    template = PromptTemplate(
        template="Give me a brief summary of topic : {topic}", input_variables=["topic"]
    )
    res = llm.invoke(template.format(topic=topic))
    return res


# This is also correct
"""
chain = RunnableLambda(lambda x: "blockchain") | RunnableParallel(
    {
        "brief": RunnableLambda(getBriefReport) | parser,
        "short_summary": RunnableLambda(getShortSummary) | parser,
    }
)
"""

chain = RunnableSequence(
    RunnableLambda(lambda x: "blockchain"),
    RunnableParallel(
        {
            "brief": RunnableSequence(RunnableLambda(getBriefReport), parser),
            "short_summary": RunnableSequence(RunnableLambda(getShortSummary), parser),
        }
    ),
)

res = chain.invoke({})

print(res["brief"])

print("-------------------------------------------------------------------------------")

print(res["short_summary"])
