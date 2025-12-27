# Structured Output with Json Schema
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

# Json Schema
json_schema_MyData = {
    "title": "MyData",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "write the summary of the article",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "write the key points of the article",
        },
        "author_name": {
            "type": ["string", "null"],
            "description": "write the name of the author",
        },
    },
    "required": ["summary", "key_points"],
}

llm = ChatOpenAI(model="gpt-3.5-turbo")
structured_llm = llm.with_structured_output(json_schema_MyData)

query = "This article explains how daily exercise improves mental and physical health, highlighting benefits like increased energy, reduced stress, and better sleep"

res = structured_llm.invoke(query)
res = dict(res)
print(res["summary"])
