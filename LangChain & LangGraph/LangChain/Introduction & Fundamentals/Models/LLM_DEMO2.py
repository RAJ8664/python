# OpenSource Model from HuggingFace using Inference API
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
)

res = model.invoke("what is the capital of Nepal?")

print(res.content)
