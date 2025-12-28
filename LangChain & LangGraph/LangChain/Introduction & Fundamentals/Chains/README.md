## 1. Introduction to Chains

**Chains** are considered the most fundamental component of the framework—so much so that the name "LangChain" is derived from them. While previous modules focused on individual components like Models and Prompts, Chains are the mechanism used to **link these components together** to build functional applications.

### The Need for Chains

Building an LLM application typically involves multiple small steps:

1.  Requesting/formatting a prompt.
2.  Sending that prompt to an LLM.
3.  Processing the LLM's response (e.g., extracting the text from metadata).

Executing these steps **manually** (calling `invoke()` separately for each part) is tedious and inefficient for complex applications. **Chains automate this by creating a pipeline** where the output of one step automatically becomes the input for the next.

---

## 2. LangChain Expression Language (LCEL)

The modern way to build chains is through **LCEL**, which uses a **declarative syntax**.

- **Pipe Operator (`|`):** Components are joined using the pipe symbol, similar to Unix piping.
- **Syntax Example:** `chain = prompt | model | parser`.
- **Benefits:** It simplifies code significantly; what previously took multiple lines of manual data handling can now be executed in a single trigger.

---

## 3. Types of Chains

### A. Sequential Chains

In a sequential chain, steps are executed one after another in a linear series.

- **Simple Example:** A chain that takes a topic, generates five facts using an LLM, and parses the result into a string.
- **Complex Example:** A "long" chain where the output of one LLM call serves as the basis for a second call. For instance, a topic is sent to an LLM to generate a **detailed report**, and that report is then passed back into the LLM with a new prompt to extract a **five-pointer summary**.

### B. Parallel Chains

**Parallel Chains** allow you to execute multiple chains simultaneously from a single input.

- **RunnableParallel:** This class is used to run several operations at once.
- **Use Case Example:** A user provides a technical document. The system uses **two different models parallelly**: one generates **study notes** while the other generates a **quiz**.
- **Merging:** After the parallel tasks finish, a final chain can "merge" their outputs into a single document.

### C. Conditional Chains

**Conditional Chains** use "if-else" logic to decide which path the execution should take based on the LLM's output.

- **RunnableBranch:** This utility acts as the "if-else" statement of the LangChain universe.
- **The Logic:** You provide a list of tuples: `(condition, chain_to_execute)`. If no conditions are met, a default chain runs.
- **Use Case Example (Sentiment Analysis):**
    1.  **Classifier:** An LLM determines if a customer's feedback is "Positive" or "Negative".
    2.  **Branching:** If positive, the "Positive Response Chain" triggers; if negative, the "Negative Response Chain" triggers to handle the complaint.
- **Requirement for Consistency:** Because branching depends on exact words (like "Positive"), it is crucial to use **Pydantic/Structured Output** to ensure the classifier returns a consistent label rather than a conversational sentence.

---

## 4. Key Utilities and Methods

- **RunnableLambda:** A utility that converts a standard Python function (like a lambda) into a "Runnable" component that can be part of a chain.
- **Visualisation:** You can view the structure of any chain using the `get_graph().print_ascii()` method. This generates a visual map showing how data flows from the prompt through the models and parsers.
