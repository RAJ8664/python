# Example of embeddings in langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

res = llm.embed_query("what is the capital of Nepal?")

print(str(res))
