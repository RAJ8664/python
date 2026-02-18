from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=0)

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="DB",
    collection_name="temp_file_embedding",
)

docs = TextLoader("temp.txt").load()

docs = splitter.split_documents(
    docs
)  # List of Documents(page_content, metadata) --> splitted the content from temp.txt

vector_store.add_documents(docs)

temp = dict(vector_store.get(include=["embeddings", "documents"]))

# print(temp)

ids = temp["ids"]
embeddings = temp["embeddings"]
documents = temp["documents"]

for i in range(len(ids)):
    print(f"id : {ids[i]}, document : {documents[i]}, embeddings: {embeddings[i]}")

similar_doc = vector_store.similarity_search(query="Cricket", k=1)

print(similar_doc)


"""
Output: 

id : 3ca2e214-f206-4e10-8606-a46c08c846a8, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : d7d582ac-dfb5-44b6-9368-a8b17368aece, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : ac593326-46a7-4489-85c9-f87edbc29635, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
id : 40ba0959-02b2-45f6-9995-d6494822ffdf, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : 673904b6-a55a-41d5-8d78-5509e40c7802, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : a868c383-9f54-4b6f-a9b3-f4a3a778f6e7, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
id : 051129dc-9f65-4657-b93a-d35301fbfe0a, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : dd905354-2796-4aba-b856-f02af6aab1cc, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : 40f2fb58-8a50-46ce-a2ab-3701d3552504, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
id : ddc293c8-0c7c-4e29-ad90-bbdf7bd52f96, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : ee0fd3eb-b1ba-45f5-8f37-2fed3b790a19, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : 0046fcc3-f470-489c-ad35-3cdac194a2d5, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
id : 80e03195-05af-4452-b65b-83dc0e4d4a40, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : 84a76371-0f21-4bfc-bc94-8829514be575, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : 4517c652-6e39-4092-94c8-2b6e2c2c5dcd, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
id : 289b420a-1239-4ff1-a419-1765f9e4ab77, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : 78efa6b8-7e44-4c1f-8996-1146d06929eb, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : aea6c21b-946f-4034-a6e3-1538135ffa0f, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
id : 2753bc4a-d316-46cd-b983-c8b2489da939, document : Virat Kohli is and indian cricketer., embeddings: [-0.01895999  0.00565998  0.01237324 ...  0.00579426 -0.01013077
 -0.00276831]
id : 035d3c04-7a50-4ea0-803c-2f511f578fc9, document : Lionel Messi is a footballer., embeddings: [-0.02021546  0.01506819  0.01612376 ...  0.01409858  0.00790826
  0.00200508]
id : 75bba110-7d47-462f-98b1-d181be56bbb3, document : Arijit Singh is a singer., embeddings: [-0.02728624  0.00632703  0.00159648 ...  0.00850758 -0.00611271
 -0.00648525]
[Document(id='80e03195-05af-4452-b65b-83dc0e4d4a40', metadata={'source': 'temp.txt'}, page_content='Virat Kohli is and indian cricketer.')]

"""
