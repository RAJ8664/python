# LangChain & LangGraph

A comprehensive guide to building LLM-powered applications using LangChain and LangGraph frameworks.

## Table of Contents

- [Overview](#overview)
- [What is LangChain?](#what-is-langchain)
- [What is LangGraph?](#what-is-langgraph)
- [Key Differences](#key-differences)
- [Architecture Comparison](#architecture-comparison)
- [Installation](#installation)
- [LangChain Fundamentals](#langchain-fundamentals)
- [LangGraph Fundamentals](#langgraph-fundamentals)
- [Use Cases](#use-cases)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Overview

This repository contains everything you need to understand and implement LLM applications using LangChain and LangGraph. Both frameworks are developed by LangChain AI to simplify building applications powered by Large Language Models (LLMs).

**Quick Summary:**

- **LangChain**: Framework for building LLM applications with chains, prompts, and tools
- **LangGraph**: Extension of LangChain for building stateful, multi-actor applications with cyclic graphs

---

## What is LangChain?

LangChain is a framework designed to simplify the creation of applications using large language models. It provides modular components that can be chained together to create complex workflows.

### Core Concepts

**1. Chains**

- Sequential pipelines that connect multiple components
- Data flows from one component to the next
- Example: Prompt → LLM → Output Parser

**2. Prompts**

- Templates for structuring input to LLMs
- Support for dynamic variables and few-shot examples
- Prompt management and versioning

**3. Models**

- Integration with various LLM providers (OpenAI, Anthropic, etc.)
- Unified interface for different models
- Support for chat models and completion models

**4. Memory**

- Conversation history management
- Context retention across interactions
- Different memory types (Buffer, Summary, Entity)

**5. Agents**

- LLMs that can use tools to accomplish tasks
- Dynamic decision-making based on user input
- ReAct (Reasoning + Acting) framework

**6. Tools**

- External capabilities the LLM can invoke
- Examples: Web search, calculators, databases
- Custom tool creation

### LangChain Components Diagram

```
┌─────────────────────────────────────────────┐
│           LangChain Application             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐    ┌──────────┐               │
│  │  Prompt  │───▶│   LLM    │               │
│  │ Template │    │  Model   │               │
│  └──────────┘    └─────┬────┘               │
│                        │                    │
│                        ▼                    │
│                  ┌──────────┐               │
│                  │  Output  │               │
│                  │  Parser  │               │
│                  └──────────┘               │
│                                             │
│  ┌──────────────────────────────┐           │
│  │         Memory               │           │
│  │  (Conversation History)      │           │
│  └──────────────────────────────┘           │
│                                             │
│  ┌──────────────────────────────┐           │
│  │          Tools               │           │
│  │  • Web Search                │           │
│  │  • Calculator                │           │
│  │  • Database                  │           │
│  └──────────────────────────────┘           │
└─────────────────────────────────────────────┘
```

---

## What is LangGraph?

LangGraph is a library built on top of LangChain for creating stateful, multi-actor applications with LLMs. It extends LangChain by adding support for cycles, controllability, and persistence through graph-based workflows.

### Core Concepts

**1. State Graph**

- Directed graph where nodes are functions
- Edges define the flow between nodes
- State is passed and modified through the graph

**2. Nodes**

- Individual functions that process state
- Can be LLM calls, tool executions, or custom logic
- Each node receives and returns the state

**3. Edges**

- Connect nodes in the graph
- Can be conditional (based on state values)
- Support for cycles and loops

**4. State**

- Central data structure passed through the graph
- Can contain messages, variables, and context
- Persisted across graph executions

**5. Checkpointing**

- Automatic state persistence
- Ability to resume from any point
- Time-travel debugging capabilities

**6. Human-in-the-Loop**

- Pause execution for human input
- Approval workflows
- Interactive debugging

### LangGraph Architecture Diagram

```
┌────────────────────────────────────────────┐
│         LangGraph Application              │
├────────────────────────────────────────────┤
│                                            │
│              ┌──────────┐                  │
│              │  START   │                  │
│              └────┬─────┘                  │
│                   │                        │
│                   ▼                        │
│             ┌─────────┐                    │
│         ┌───│  Agent  │───┐                │
│         │   │  Node   │   │                │
│         │   └─────────┘   │                │
│         │                 │                │
│         ▼                 ▼                │
│    ┌─────────┐      ┌──────────┐           │
│    │  Tool   │      │Continue? │           │
│    │  Node   │      │(Conditional)         │
│    └────┬────┘      └────┬─────┘           │
│         │                │                 │
│         │                │ Yes             │
│         └────────┬───────┘                 │
│                  │                         │
│                  │ No                      │
│                  ▼                         │
│             ┌─────────┐                    │
│             │   END   │                    │
│             └─────────┘                    │
│                                            │
│  ┌──────────────────────────────┐          │
│  │      Persistent State        │          │
│  │    (Checkpointing Layer)     │          │
│  └──────────────────────────────┘          │
└────────────────────────────────────────────┘
```

---

## Key Differences

| Feature               | LangChain                      | LangGraph                     |
| --------------------- | ------------------------------ | ----------------------------- |
| **Structure**         | Linear chains/sequential flows | Cyclic graphs with loops      |
| **State Management**  | Limited, through Memory        | Built-in stateful workflows   |
| **Control Flow**      | Predetermined sequence         | Dynamic, conditional routing  |
| **Persistence**       | Manual implementation          | Automatic checkpointing       |
| **Debugging**         | Standard logging               | Time-travel debugging         |
| **Human-in-the-Loop** | Manual integration             | Native support                |
| **Use Case**          | Simple, sequential tasks       | Complex, multi-step workflows |
| **Learning Curve**    | Easier for beginners           | Requires understanding graphs |

### When to Use What?

**Use LangChain when:**

- Building simple question-answering systems
- Creating basic chatbots
- Implementing straightforward RAG applications
- Working with predetermined workflows
- Getting started with LLM applications

**Use LangGraph when:**

- Building complex agent systems
- Need multiple decision points
- Require state persistence across sessions
- Implementing human-in-the-loop workflows
- Building multi-agent systems
- Need fine-grained control over execution

---

## Architecture Comparison

### LangChain Flow (Linear)

```
User Input → Prompt → LLM → Parser → Output
     ↓
  Memory
```

### LangGraph Flow (Cyclic)

```
          ┌─────────────────┐
          │   User Input    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Route Agent    │◄───┐
          └────────┬────────┘    │
                   │             │
         ┌─────────┴─────────┐   │
         ▼                   ▼   │
   ┌──────────┐        ┌──────────┐
   │  Tool A  │        │  Tool B  │
   └─────┬────┘        └─────┬────┘
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Should Loop?   │
          └────────┬────────┘
                   │
         Yes ──────┘     No
                          │
                          ▼
                    ┌──────────┐
                    │  Output  │
                    └──────────┘
```

---

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version
```

### Install LangChain

```bash
# Core LangChain
pip install langchain

# LangChain Community (additional integrations)
pip install langchain-community

# LangChain OpenAI
pip install langchain-openai

# LangChain Anthropic
pip install langchain-anthropic
```

### Install LangGraph

```bash
# LangGraph
pip install langgraph

# With checkpointing support
pip install langgraph[checkpointing]
```

### Additional Dependencies

```bash
# For vector stores
pip install chromadb faiss-cpu

# For document loaders
pip install pypdf docx2txt

# For web scraping
pip install beautifulsoup4 playwright
```

---

## LangChain Fundamentals

### Basic Chain Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize components
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

# Create chain
chain = prompt | llm | output_parser

# Execute
result = chain.invoke({"input": "What is LangChain?"})
print(result)
```

### Agent with Tools Example

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# Setup
llm = ChatOpenAI(model="gpt-4")
tools = [get_word_length]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Execute
result = agent_executor.invoke({"input": "How many letters are in 'LangChain'?"})
print(result)
```

### RAG (Retrieval Augmented Generation) Example

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load and split documents
loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# Create retrieval chain
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the context below:

Context: {context}

Question: {question}

Answer:"""
)

# Query
question = "What is the main topic?"
docs = retriever.get_relevant_documents(question)
context = "\n\n".join([doc.page_content for doc in docs])

chain = prompt | llm
result = chain.invoke({"context": context, "question": question})
print(result.content)
```

---

## LangGraph Fundamentals

### Basic Graph Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class State(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

# Define nodes
def node_1(state: State) -> State:
    return {"messages": ["Node 1 executed"], "count": state["count"] + 1}

def node_2(state: State) -> State:
    return {"messages": ["Node 2 executed"], "count": state["count"] + 1}

# Build graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("node_1", node_1)
workflow.add_node("node_2", node_2)

# Add edges
workflow.set_entry_point("node_1")
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", END)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({"messages": [], "count": 0})
print(result)
```

### Agent with Conditional Routing

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: list
    next: str

def call_agent(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState) -> AgentState:
    # Simulate tool execution
    tool_response = "Tool executed successfully"
    return {"messages": [AIMessage(content=tool_response)]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    # Simple logic: if message contains "tool", use tools
    if "tool" in last_message.content.lower():
        return "tools"
    return "end"

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_agent)
workflow.add_node("tools", call_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Execute
result = app.invoke({
    "messages": [HumanMessage(content="Use a tool to check the weather")]
})
print(result)
```

### Human-in-the-Loop Example

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict

class State(TypedDict):
    input: str
    approved: bool
    output: str

def process_input(state: State) -> State:
    return {"output": f"Processing: {state['input']}"}

def human_approval(state: State) -> State:
    # This will pause for human input
    print(f"Approve this action: {state['output']}? (yes/no)")
    return state

def route_after_approval(state: State) -> str:
    return "finalize" if state.get("approved") else END

# Build graph with checkpointing
memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(State)

workflow.add_node("process", process_input)
workflow.add_node("approval", human_approval)
workflow.add_node("finalize", lambda s: {"output": "Finalized!"})

workflow.set_entry_point("process")
workflow.add_edge("process", "approval")
workflow.add_conditional_edges("approval", route_after_approval)
workflow.add_edge("finalize", END)

app = workflow.compile(checkpointer=memory)
```

---

## Use Cases

### LangChain Use Cases

1. **Chatbots & Virtual Assistants**
    - Customer service bots
    - FAQ answering systems
    - Personal assistants

2. **Document Q&A**
    - Search through documentation
    - Legal document analysis
    - Research paper summarization

3. **Content Generation**
    - Blog post writing
    - Marketing copy creation
    - Code documentation

4. **Data Extraction**
    - Information extraction from text
    - Form filling automation
    - Entity recognition

5. **Translation & Summarization**
    - Multi-language translation
    - Meeting notes summarization
    - Article summarization

### LangGraph Use Cases

1. **Complex Agent Systems**
    - Multi-step research agents
    - Planning and execution agents
    - Self-correcting agents

2. **Multi-Agent Collaboration**
    - Team of specialized agents
    - Debate and consensus systems
    - Hierarchical agent structures

3. **Human-in-the-Loop Workflows**
    - Approval workflows
    - Content moderation systems
    - Interactive decision making

4. **Stateful Applications**
    - Long-running conversations
    - Task tracking systems
    - Session management

5. **Cyclic Workflows**
    - Iterative refinement processes
    - Code generation and testing loops
    - Research and validation cycles

---

## Examples

### Example 1: Simple Chatbot (LangChain)

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Setup
llm = ChatOpenAI(temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Chat
print(conversation.predict(input="Hi, my name is Alice"))
print(conversation.predict(input="What's my name?"))
```

### Example 2: Research Agent (LangGraph)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class ResearchState(TypedDict):
    topic: str
    research_steps: List[str]
    findings: List[str]
    final_report: str

def plan_research(state: ResearchState) -> ResearchState:
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"Create a research plan for: {state['topic']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    steps = response.content.split("\n")
    return {"research_steps": steps}

def conduct_research(state: ResearchState) -> ResearchState:
    findings = []
    for step in state["research_steps"][:3]:  # Limit to 3 steps
        finding = f"Research completed for: {step}"
        findings.append(finding)
    return {"findings": findings}

def compile_report(state: ResearchState) -> ResearchState:
    report = f"Research Report on {state['topic']}\n\n"
    report += "\n".join(state["findings"])
    return {"final_report": report}

# Build graph
workflow = StateGraph(ResearchState)
workflow.add_node("plan", plan_research)
workflow.add_node("research", conduct_research)
workflow.add_node("report", compile_report)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "research")
workflow.add_edge("research", "report")
workflow.add_edge("report", END)

app = workflow.compile()

# Execute
result = app.invoke({"topic": "Climate change impacts"})
print(result["final_report"])
```

### Example 3: Code Generation with Validation (LangGraph)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class CodeState(TypedDict):
    requirements: str
    code: str
    tests_passed: bool
    iteration: int

def generate_code(state: CodeState) -> CodeState:
    # Simulate code generation
    code = f"def solution():\n    # Generated for: {state['requirements']}\n    pass"
    return {"code": code}

def test_code(state: CodeState) -> CodeState:
    # Simulate testing
    tests_passed = state["iteration"] >= 2  # Pass after 2 iterations
    return {"tests_passed": tests_passed, "iteration": state["iteration"] + 1}

def should_regenerate(state: CodeState) -> Literal["generate", "end"]:
    if not state["tests_passed"] and state["iteration"] < 3:
        return "generate"
    return "end"

# Build graph
workflow = StateGraph(CodeState)
workflow.add_node("generate", generate_code)
workflow.add_node("test", test_code)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "test")
workflow.add_conditional_edges(
    "test",
    should_regenerate,
    {"generate": "generate", "end": END}
)

app = workflow.compile()

# Execute
result = app.invoke({
    "requirements": "Sort a list of numbers",
    "iteration": 0
})
print(f"Final code:\n{result['code']}")
print(f"Tests passed: {result['tests_passed']}")
```

---

## Best Practices

### LangChain Best Practices

1. **Use Prompt Templates**
    - Keep prompts organized and reusable
    - Version control your prompts
    - Use few-shot examples when needed

2. **Implement Error Handling**
    - Catch API failures gracefully
    - Implement retry logic
    - Set timeouts appropriately

3. **Optimize Token Usage**
    - Use cheaper models for simple tasks
    - Implement caching when possible
    - Trim conversation history

4. **Security**
    - Never expose API keys
    - Sanitize user inputs
    - Implement rate limiting

5. **Testing**
    - Test with various inputs
    - Monitor LLM responses
    - Implement evaluation metrics

### LangGraph Best Practices

1. **State Design**
    - Keep state structure simple
    - Use TypedDict for type safety
    - Document state transitions

2. **Node Functions**
    - Keep nodes focused and single-purpose
    - Return partial state updates
    - Handle errors within nodes

3. **Checkpointing**
    - Use persistent storage for production
    - Implement cleanup for old checkpoints
    - Test resume functionality

4. **Conditional Routing**
    - Make routing logic clear and testable
    - Avoid complex conditions
    - Document decision points

5. **Debugging**
    - Use visualization tools
    - Log state at each step
    - Test individual nodes separately

---

## Resources

### Official Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain API Reference](https://api.python.langchain.com/)

### Community

- [LangChain Discord](https://discord.gg/langchain)
- [GitHub - LangChain](https://github.com/langchain-ai/langchain)
- [GitHub - LangGraph](https://github.com/langchain-ai/langgraph)

### Tutorials & Courses

- LangChain YouTube Channel
- DeepLearning.AI - LangChain Courses
- LangChain Blog

### Related Tools

- **LangSmith**: Platform for debugging and monitoring
- **LangServe**: Deploy LangChain applications
- **LangChain Hub**: Share and discover prompts

---

## Getting Started Checklist

- [ ] Install LangChain and LangGraph
- [ ] Set up API keys for LLM providers
- [ ] Run basic LangChain example
- [ ] Build your first LangGraph workflow
- [ ] Explore advanced examples
- [ ] Build your own application
