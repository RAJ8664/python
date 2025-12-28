from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

parser = StrOutputParser()

template1 = PromptTemplate(
    template="write a detailed report on the topic : {topic}", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write a summary of the given text : \n {text}",
    input_variables=["text"],
)

chain = template1 | llm | parser | template2 | llm | parser

res = chain.invoke({"topic": "BlockChain"})

print(res)

# Print the chain
# chain.get_graph().print_ascii()
