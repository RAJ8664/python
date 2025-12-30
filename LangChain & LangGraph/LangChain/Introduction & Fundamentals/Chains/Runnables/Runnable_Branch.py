from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the feedback"
    )


parser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback:\n {feedback} \n {format}",
    input_variables=["feedback"],
    partial_variables={"format": parser.get_format_instructions()},
)


chain = RunnableSequence(
    prompt,
    llm,
    parser,
    RunnableBranch(
        (lambda x: x.sentiment == "positive", RunnableLambda(lambda x: "Positive")),
        (lambda x: x.sentiment == "negative", RunnableLambda(lambda x: "Negative")),
        (lambda x: x.sentiment == "neutral", RunnableLambda(lambda x: "Neutral")),
        RunnableLambda(lambda x: "Could not find sentiment"),
    ),
)

res = chain.invoke({"feedback": "This is a good product"})
print(res)
