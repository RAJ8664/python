from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = []

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    if len(user_input) == 0:
        continue
    chat_history.append(user_input)
    res = llm.invoke(chat_history)
    print("AI : ", res.content)
    chat_history.append(res.content)

print(chat_history)

# Issue with this version
"""
Our list has all the chats history of bothe user and AI 
we want to seperate this somehow (may be using dictionary [key : value, key --> user | ai, value --> message])
This is where LangChain inbuilt thing comes in picture called as messages 
There are three types of messages in langchain
1. System Messages 
2. Human Messages
3. AI Messages
Example is shown in ChatbotV3.py

"""
