# File where i practiced fundamentals of python 🫣


import sys

sys.path.append(
    "/home/rkroy/Desktop/code/python/LangChain & LangGraph/LangChain/Introduction & Fundamentals/Models"
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
