from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph, state
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"))


@tool
def add(a: int, b: int) -> int:
    """
    Use this tool to add two numbers.

    Args:
        a: The first number to add.
        b: The second number to add.

    Returns:
        The sum of the two numbers.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    Use this tool to multiply two numbers.

    Args:
        a: The first number to multiply.
        b: The second number to multiply.

    Returns:
        The product of the two numbers.
    """
    return a * b


@tool
def subtract(a: int, b: int) -> int:
    """
    Use this tool to subtract two numbers.

    Args:
        a: The first number to subtract.
        b: The second number to subtract.

    Returns:
        The difference of the two numbers.
    """
    return a - b


tools = [subtract, multiply, add]
model_with_tools = model.bind_tools(tools)
