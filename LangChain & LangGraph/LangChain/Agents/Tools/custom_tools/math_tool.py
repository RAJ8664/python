# NOTE: There are multiple ways to create tools in langchain (refer docs)

from langchain.tools import BaseTool
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import math
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import Type
from langchain.tools import ToolRuntime  # TODO: checkout docs

load_dotenv()


class Circle(BaseModel):
    radius: float | int = Field(description="radius of the circle")


class CircleAreaTool(BaseTool):
    name: str = "Circle Area Tool"
    description: str = (
        "Use this to when you need to find or calculate the area of circle."
    )
    args_schema: Type[BaseModel] = Circle

    def _run(self, radius: float | int) -> float:
        """
        Calculates the area of a circle.
        """
        return float(math.pi * radius**2)

    def _arun(self, radius: float | int):
        raise NotImplementedError("This tool does not support asynchronous execution.")


custom_tools = [CircleAreaTool()]
model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"), temperature=0
)

agent = create_agent(model=model, tools=custom_tools)

events = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the area of a circle with a radius of 5?",
            }
        ]
    },
    stream_mode="values",
)

conversation = list(events["messages"])
for event in conversation:
    print(event.content)

print("Simple conversation ended")

"""
Output: 
What is the area of a circle with a radius of 5?
I'll calculate the area of a circle with radius 5 for you.


78.53981633974483
The area of a circle with radius 5 is approximately **78.54 square units** (using π ≈ 3.14159, the exact value is 25π or about 78.5398).
"""
print("Full conversation started")

# If you want complete conversation (user stream instead of invoke)
full_events = agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the area of a circle with a radius of 5?",
            }
        ]
    },
    stream_mode="values",
)

full_conversation = []
for events in full_events:
    full_conversation.append(events["messages"][-1].pretty_print())

for event in full_conversation:
    print(event)

"""
Output: 

Full conversation started
================================ Human Message =================================

What is the area of a circle with a radius of 5?
================================== Ai Message ==================================

I'll calculate the area of a circle with a radius of 5 using the Circle Area Tool.
Tool Calls:
  Circle Area Tool (667e3ab8e297470683b3bde359020f7d)
 Call ID: 667e3ab8e297470683b3bde359020f7d
  Args:
    radius: 5
================================= Tool Message =================================
Name: Circle Area Tool

78.53981633974483
================================== Ai Message ==================================

The area of a circle with a radius of 5 is approximately **78.54** square units.
"""
