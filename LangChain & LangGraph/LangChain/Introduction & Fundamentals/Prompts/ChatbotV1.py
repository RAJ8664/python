# Basic ChatBot in LangChain (Console Based)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    if len(user_input) == 0:
        continue
    res = llm.invoke(user_input)
    print("AI : ", res.content)

# Issue with this version
""" 
You : Hi 
AI :  Hi there! How can I help you today?
You : which one is greater 2 or 0 ? 
AI :  **2** is greater than 0.
You : now multiply it by 10
AI : Ai will not understand with which number it is supposed to multiply by 10. (No context)

--> To Overcome this issue, we came make use of list
--> Example shown in ChatbotV2.py

"""
