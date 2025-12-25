from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Got quota limit reached 😢
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Not an issue, will find another way

# This is kind of chat history
messages = []

while True:
    user_input = input("You : ")
    if user_input == "exit":
        break
    if len(user_input) == 0:
        continue
    messages.append(HumanMessage(content=user_input))
    res = llm.invoke(messages)
    messages.append(AIMessage(content=res.content))
    print("AI : ", res.content)

print(messages)
