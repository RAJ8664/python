from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

wiki_retriever = WikipediaRetriever()

docs = wiki_retriever.invoke("Kathmandu Nepal")

for doc in docs:
    with open("retrieved_data.txt", mode="w") as file:
        file.write(doc.page_content)


with open("retrieved_data.txt", mode="r") as file:
    for line in file:
        print(line)
