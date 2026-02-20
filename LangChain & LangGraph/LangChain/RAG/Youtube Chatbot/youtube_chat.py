from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnableParallel,
)
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

chat_model = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0.7
)  # Note the model we are using, its so old. Imagine if it were something else(Good one).

embedding_model = OpenAIEmbeddings()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

vector_store = Chroma(
    persist_directory="DB",
    collection_name="youtube_data",
    embedding_function=embedding_model,
)

str_parser = StrOutputParser()

youtube_doc = YoutubeLoader(
    video_id="Z8eXaXoUJRQ", language="hi"
).load()  # Transcript for video song named --> slow down let's see what happens
youtube_doc = splitter.split_documents(youtube_doc)

# file = open("transcript.txt", "w")
# for doc in youtube_doc:
#     file.write(doc.page_content)
# file.close()

vector_store.add_documents(youtube_doc)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

while True:
    query = input("Enter the question Regarding this video: ")

    if query == "exit":
        break
    got = retriever.invoke(query)[0].page_content

    prompt = PromptTemplate(
        template="Based on the content i am providing you ansewr the following query according to the content \n content : {content} \n query : {query}",
        input_variables=["content", "query"],
    )

    chain = prompt | chat_model | str_parser

    res = chain.invoke({"content": got, "query": query})

    print(res)
