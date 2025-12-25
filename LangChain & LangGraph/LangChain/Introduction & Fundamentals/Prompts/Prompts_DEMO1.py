# Basic Example to demonstrate LangChain usage for PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Define the base prompt string with a placeholder.
template = "Tell me about footballer {name}"

# Create a reusable prompt template object.
prompt = PromptTemplate.from_template(template)

# Populate the prompt with a specific value.
formatted_prompt = prompt.format(name="Lionel Messi")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

res = llm.invoke(formatted_prompt)

print(res.content)
