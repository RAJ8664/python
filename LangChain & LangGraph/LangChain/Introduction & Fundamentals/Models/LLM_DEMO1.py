# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

res = llm.invoke("Can you write me a segment tree code in java")
print(res.content)
