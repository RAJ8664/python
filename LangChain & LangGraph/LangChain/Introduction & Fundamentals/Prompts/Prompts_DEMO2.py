# Example of streamlit
# To Run --> streamlit run Prompts_DEMO2.py

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("ChatBot")

user_prompt = st.text_input("Enter your Prompt Here")

if st.button("Submit"):
    res = llm.invoke(user_prompt)
    st.write(res.content)
