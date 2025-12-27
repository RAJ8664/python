# Demo using TypedDict

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()


class MyData(TypedDict):
    summary: Annotated[str, "Describe the summary"]
    key_points: Annotated[list[str], "List the key points"]
    author_name: Annotated[
        Optional[str], "Name of the author"
    ]  # It is not necessary that the article will always have an author (For example)


llm = ChatOpenAI(model="gpt-3.5-turbo")
structured_llm = llm.with_structured_output(MyData)

query = "This article explains how daily exercise improves mental and physical health, highlighting benefits like increased energy, reduced stress, and better sleep. The piece is written by Dr. Ananya Sharma"

res = structured_llm.invoke(query)
print(res["summary"])

"""

Here the issue is , even thought we have mentioned that the data type of summary should be string, it is not necessary that the response returned by model will be string. 
we cannot apply data type checking here.
To Fix this issue, we can use pydantic -> explained in Demo2.py

"""
