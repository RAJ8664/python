# Exmaple : Pydantic Output Parser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")


class MyData(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=MyData)

template = PromptTemplate(
    template="write info of a random person in Nepal \n {output_format}",
    input_variables=[],
    partial_variables={"output_format": parser.get_format_instructions()},
)

chain = template | llm | parser
res = chain.invoke({})
print(res)

print()


# What if i want to have list of such people
class MyData2(BaseModel):
    people: list[MyData] = Field(description="List of people")


parser2 = PydanticOutputParser(pydantic_object=MyData2)

template2 = PromptTemplate(
    template="write info of 10 random person in Nepal \n {output_format}",
    input_variables=[],
    partial_variables={"output_format": parser2.get_format_instructions()},
)

chain2 = template2 | llm | parser2
res2 = chain2.invoke({})
print(res2)
