## 1. The Evolution and Purpose of Runnables

In the early stages of LangChain (post-2022), the framework provided various components like **PromptTemplates**, **LLMs**, and **Output Parsers**. However, these components were not standardised; they used different methods to interact (e.g., `format()` for prompts, `predict()` for LLMs, `parse()` for parsers). This made it difficult for developers to connect them flexibly.

Initially, the LangChain team created hardcoded **"Chains"** (like `LLMChain` or `RetrievalQA`) to automate these connections. While helpful, this led to a "bloated" library with too many specific chains, making the learning curve steep for new engineers.

The solution was the **Runnable**—an abstract base class that standardises all components so they can be "plugged and played" like **Lego blocks**.

### The Four Principles of Runnables:

1.  **Unit of Work:** Every runnable performs a specific task (takes input, processes, returns output).
2.  **Common Interface:** Every runnable implements standard methods: `invoke()` (single input), `batch()` (multiple inputs), and `stream()`.
3.  **Connectable:** Runnables can be linked so the output of one becomes the input of the next.
4.  **Composition:** A chain made of runnables is itself a runnable.

---

## 2. Implementing Runnables "From Scratch"

### A. The Standard Interface (Abstract Class)

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass
```

### B. Defining Components (Task-Specific Runnables)

These classes inherit from the `Runnable` base class, forcing them to use the `invoke` method.

```python
import random

class FakeLLM(Runnable):
    def invoke(self, prompt):
        responses = ["Delhi is the capital of India", "IPL is a cricket league", "AI is powerful"]
        return {"response": random.choice(responses)}

class FakePromptTemplate(Runnable):
    def __init__(self, template):
        self.template = template
    def invoke(self, input_dict):
        return self.template.format(**input_dict)

class FakeStrParser(Runnable):
    def invoke(self, input_dict):
        return input_dict["response"]
```

### C. The Connector (Building the Chain)

A "Connector" iterates through a list of runnables, passing the output of the previous one to the next.

```python
class RunnableConnector(Runnable):
    def __init__(self, runnables):
        self.runnables = runnables
    def invoke(self, input_data):
        for runnable in self.runnables:
            input_data = runnable.invoke(input_data)
        return input_data

# Usage:
template = FakePromptTemplate("Tell me about {topic}")
model = FakeLLM()
parser = FakeStrParser()

chain = RunnableConnector([template, model, parser])
print(chain.invoke({"topic": "India"}))
```

---

## 3. Categories of Runnables

1.  **Task-Specific Runnables:** Core components like `ChatOpenAI`, `PromptTemplate`, or `StrOutputParser`.
2.  **Runnable Primitives:** Building blocks used to orchestrate how tasks interact (Sequence, Parallel, etc.).

---

## 4. Key Runnable Primitives

### A. Runnable Sequence

Connects runnables linearly. The output of R1 goes to R2.

- **Example Code:**
    ```python
    from langchain_core.runnables import RunnableSequence
    chain = RunnableSequence([prompt, model, parser])
    result = chain.invoke({"topic": "AI"})
    ```

### B. Runnable Parallel

Executes multiple runnables simultaneously on the same input. Returns a dictionary of results.

- **Example Code:**
    ```python
    from langchain_core.runnables import RunnableParallel
    # Generates a tweet and a LinkedIn post at the same time
    parallel_chain = RunnableParallel(
        tweet=prompt1 | model | parser,
        linkedin=prompt2 | model | parser
    )
    result = parallel_chain.invoke({"topic": "Generative AI"})
    ```

### C. Runnable PassThrough

Acts as a placeholder that passes the input directly to the output without modification. It is vital for maintaining data in a parallel chain.

- **Use Case:** If you generate a joke and want to output both the joke and its explanation, use `PassThrough` to keep the joke string for the final output.

### D. Runnable Lambda

Converts any custom Python function into a runnable so it can be used in a chain.

- **Example Code:**

    ```python
    from langchain_core.runnables import RunnableLambda
    def count_words(text):
        return len(text.split())

    word_count_runnable = RunnableLambda(count_words)
    # Now it can be piped: model | parser | word_count_runnable
    ```

### E. Runnable Branch

Implements "if-else" logic. It takes a list of (condition, runnable) tuples and a default runnable.

- **Use Case:** Summarise a report only if it exceeds 500 words; otherwise, return it as is.
- **Syntax:** `RunnableBranch((condition_lambda, true_runnable), default_runnable)`.

---

## 5. LangChain Expression Language (LCEL)

LCEL is a declarative way to write `RunnableSequence` using the **Pipe (`|`) operator**.

- **Traditional:** `chain = RunnableSequence([prompt, model, parser])`.
- **LCEL:** `chain = prompt | model | parser`.
  This syntax is more readable and is now the industry standard for defining LangChain workflows.
