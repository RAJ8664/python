from langgraph.graph import START, END, StateGraph
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing_extensions import TypedDict
from IPython.display import Image, display
from pydantic import BaseModel, Field

load_dotenv()
model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"))
