# Simple Example for Json Scehema Based Structured Output
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# {{city1, temp1}, {city2, temp2}
MyData = {
    "title": "Temperature_Data",
    "type": "object",
    "properties": {
        "cities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name in India"},
                    "temperature": {
                        "type": "string",
                        "description": "Temperature in Celsius",
                    },
                },
                "required": ["city", "temperature"],
            },
        }
    },
    "required": ["cities"],
}

llm = ChatOpenAI(model="gpt-3.5-turbo")
structured_llm = llm.with_structured_output(MyData)

query = "List the temperatures of any 10 cities in India."

res = structured_llm.invoke(query)

print(res)
