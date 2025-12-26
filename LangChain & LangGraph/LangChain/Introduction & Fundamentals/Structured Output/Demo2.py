# Structured Output with Pydantic (Data Validation Can be done here)
# Pydantic is a data validation and data parsing library for python. It ensures that the data you work with is correct, structured, and type-safe.

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

# This can be use as well.
# class MyData(BaseModel):
#     summary: Annotated[str, "Describe the summary"]
#     key_points: Annotated[list[str], "List the key points"]
#     author_name: Annotated[
#         Optional[str], "Name of the author"
#     ]  # It is not necessary that the article will always have an author (For example)
#


class MyData(BaseModel):
    summary: str = Field(description="Describe the summary")
    key_points: list[str] = Field(description="List the key points")
    author_name: Optional[str] = Field(description="Name of the author")


llm = ChatOpenAI(model="gpt-3.5-turbo")
structured_llm = llm.with_structured_output(MyData)

query = "This article explains how daily exercise improves mental and physical health, highlighting benefits like increased energy, reduced stress, and better sleep"

res = structured_llm.invoke(query)
res = dict(res)

print(res["summary"])
