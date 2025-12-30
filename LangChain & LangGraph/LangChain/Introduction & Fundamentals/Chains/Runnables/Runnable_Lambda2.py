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


def getLength(topic: str):
    return len(topic)


def getSummary(topic: str):
    template = PromptTemplate(
        template="Give me a brief summary of topic : {topic}", input_variables=["topic"]
    )
    res = llm.invoke(template.format(topic=topic))
    return res


chain = RunnableSequence(
    RunnableLambda(lambda x: "blockchain"),
    RunnableParallel(
        {
            "length": RunnableLambda(getLength),
            "summary": RunnableLambda(getSummary),
        }
    ),
    RunnableLambda(lambda x: x["summary"]),
    parser,
)

res = chain.invoke({})
print(res)
