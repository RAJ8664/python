# Routing
# Read here --> https://docs.langchain.com/oss/python/langgraph/workflows-agents

# query bot

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts.string import PromptTemplateFormat
from langgraph.graph import START, END, StateGraph
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Literal
from IPython.display import Image, display
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

load_dotenv()
model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"))


class State(TypedDict):
    query: str
    classification: Literal["billing", "refund", "technical issue"]
    result: str


class classify(BaseModel):
    classification: Literal["billing", "refund", "general query", "technical issue"] = (
        Field(description="The classfication based on the given query")
    )


def classify_query(state: State) -> dict:
    parser = JsonOutputParser(pydantic_object=classify)
    prompt = PromptTemplate(
        template="You are a very good query classifier, Based on this query: {query} , i want you to classify the following query int format {format}",
        input_variables=["query"],
        partial_variables={"format": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    res = chain.invoke({"query": state["query"]})

    return {"classification": res["classification"]}


# Currently the function separation is not looking meaningful, but i you could imagine, each could have their own implementation type.
# Other way : you could just have a single function but you could prepare 'k' different prompts based on the classification of the query, and use the prompt according to the classification of the query.


def billing_query_fix(state: State) -> dict:
    prompt = PromptTemplate(
        template="You are a talented billing expert, Based on this query: {query}\n, i want you to help understand user query and suggest them how to fix their issues",
        input_variables=["query"],
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({"query": state["query"]})

    return {"result": res}


def refund_query_fix(state: State) -> dict:
    prompt = PromptTemplate(
        template="You are a talented analyst in solving issues related to refunding, Based on this query: {query}\n, i want you to help understand user query and suggest them how to fix their issues",
        input_variables=["query"],
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({"query": state["query"]})

    return {"result": res}


def technical_query_fix(state: State) -> dict:
    prompt = PromptTemplate(
        template="You are a talented computer science engineer and technical related query solver, Based on this query: {query}\n, i want you to help understand user query and suggest them how to fix their issues",
        input_variables=["query"],
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    res = chain.invoke({"query": state["query"]})

    return {"result": res}


# Conditional Node


def check_classification(
    state: State,
) -> Literal["billing_query_fix", "refund_query_fix", "technical_query_fix"]:
    if state["classification"] == "billing":
        print("ans : billing")
        return "billing_query_fix"
    elif state["classification"] == "refund":
        print("ans: refund_query_fix")
        return "refund_query_fix"
    print("ans: technical_query_fix")
    return "technical_query_fix"


# Create graph
graph = StateGraph(State)

# add nodes

graph.add_node("classify_query", classify_query)
graph.add_node("technical_query_fix", technical_query_fix)
graph.add_node("refund_query_fix", refund_query_fix)
graph.add_node("billing_query_fix", billing_query_fix)

graph.add_edge(START, "classify_query")
graph.add_conditional_edges("classify_query", check_classification)
graph.add_edge("technical_query_fix", END)
graph.add_edge("refund_query_fix", END)
graph.add_edge("billing_query_fix", END)

app = graph.compile()

initial_state = State({"query": "I am not able to change my profile picture"})

final_state = app.invoke(
    initial_state
)  # so its actually using technical_query_fix function

print(final_state)
