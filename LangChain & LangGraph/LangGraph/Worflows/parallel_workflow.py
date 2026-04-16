# Parallelization

from langgraph.graph import START, END, StateGraph
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing_extensions import TypedDict
from IPython.display import Image, display

load_dotenv()
model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"))


# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# Nodes
def call_llm_1(state: State) -> dict:
    """First LLM call to generate initial joke"""

    msg = model.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def call_llm_2(state: State) -> dict:
    """Second LLM call to generate story"""

    msg = model.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}


def call_llm_3(state: State) -> dict:
    """Third LLM call to generate poem"""

    msg = model.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: State) -> dict:
    """Combine the joke, story and poem into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# Invoke
initial_state = {
    "topic": "cats",
    "joke": "",
    "story": "",
    "poem": "",
    "combined_output": "",
}
state = parallel_workflow.invoke(
    State({"topic": "cats", "joke": "", "story": "", "poem": "", "combined_output": ""})
)
print(state["combined_output"])
