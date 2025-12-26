# What if the model does not support structured output? This is where the Output Parser comes in.
# Output Parser in LangChain help convert raw LLM responses into structured formats like JSON, CSV, Pydantic models and more. They ensure consistency, validation and easse of use in applications.
# Example: String Output Parser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Try to Use the model that does not support structured output for checking the power of the Output Parser.
# Obviously, Output Parsers are always going to work with the models that support structured output.
llm = ChatOpenAI(model="gpt-3.5-turbo")

template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="write a 5 line summary of the following text. \n {text}",
    input_variables=["text"],
)

"""
Approach 1

prompt1 = template1.format(topic="BlockChain")
first_res = llm.invoke(prompt1)
print(first_res.content)

print()

prompt2 = template2.format(text=first_res.content)
second_res = llm.invoke(prompt2)
print(second_res.content)

"""

# Approach 2 - Using String Output Parser and Chains

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser
res = chain.invoke({"topic": "BlockChain"})

print(res)
