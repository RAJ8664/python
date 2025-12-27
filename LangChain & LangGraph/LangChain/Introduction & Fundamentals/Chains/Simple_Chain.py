from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

template = PromptTemplate(
    template="write a short summary on topic : {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = template | llm | parser
res = chain.invoke({"topic": "Hyprland Window Manager"})

print(res)
