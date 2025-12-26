# Example: JSON Output Parser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Try to Use the model that does not support structured output for checking the power of the Output Parser.
# Obviously, Output Parsers are always going to work with the models that support structured output.
llm = ChatOpenAI(model="gpt-3.5-turbo")

parser = JsonOutputParser()

template = PromptTemplate(
    template="write info of 10 random person in Nepal \n {output_format}",
    input_variables=[],
    partial_variables={"output_format": parser.get_format_instructions()},
)

chain = template | llm | parser
res = chain.invoke({})
print(res)

# Note : We cannot define the specific structure of the output JSON we want.
# Take a look at an example of the above template, we don't know the exact properties of the json output.
# It may only mention name, age, city, or it may mention name, age, city, occupation, and so on.
# This is the biggest challenge of using this output parser.
