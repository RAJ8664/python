from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import Optional
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"))

# 1. Define the State
# 2. Create a StateGraph
# 3. Add nodes(functions) to the graph
# 4. Add edges to the graph
# 5. Compile the graph
# 6. execute the graph

# Define the State


class State(TypedDict):
    topic: str
    report: str
    summary: str


# Create a StateGraph

graph = StateGraph(State)

# Define all the nodes and add them to the graph


def generate_report(state: State) -> dict:
    prompt = PromptTemplate(
        template="write me a brief report on topic: {topic}",
        input_variables=["topic"],
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({"topic": state["topic"]})
    return {"report": res}


def generate_summary(state: State) -> dict:
    prompt = PromptTemplate(
        template="write me a complete summary of the report: {report}",
        input_variables=["report"],
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({"report": state["report"]})
    return {"summary": res}


# Add the nodes to the graph

graph.add_node(generate_report)
graph.add_node(generate_summary)

# Add the Edges to the graph (accroding to your workflow)

graph.add_edge(START, "generate_report")
graph.add_edge("generate_report", "generate_summary")
graph.add_edge("generate_summary", END)

# Compile the graph

app = graph.compile()

# Execute the graph

res = app.invoke(State({"topic": "python", "report": "", "summary": ""}))

print(f"TOPIC: \n {res['topic']}\n")
print(f"Report: \n {res['report']}\n")
print(f"Summary: \n {res['summary']}\n")
