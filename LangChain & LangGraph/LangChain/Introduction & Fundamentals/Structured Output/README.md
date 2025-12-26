## 1. Introduction to Structured Output

While Large Language Models (LLMs) are famous for communicating in natural language (text), this "unstructured" format is difficult for machines to process,. **Structured Output** refers to the practice of forcing a model to return responses in a well-defined data format, such as **JSON**,. This allows LLMs to interact directly with other systems like databases, APIs, or software tools,.

---

## 2. Key Use Cases

Structured output is essential for transforming AI from a chatbot into a functional system component:

- **Data Extraction:** Converting a resume into a database entry by extracting names, marks, and company history,.
- **API Building:** Analysing product reviews to extract specific topics (battery, display), pros/cons, and sentiment to serve via a web API (e.g., FastAPI),.
- **Building Agents:** For an AI Agent to use a tool like a calculator, it must provide specific numbers (structured data) rather than a sentence,.

---

## 3. The `with_structured_output` Function

LangChain provides a specialized function called `with_structured_output` to simplify this process.

- **How it works:** When you pass a schema to this function, LangChain generates a "System Prompt" behind the scenes that instructs the AI to act as an extractor and return data in JSON format,.
- **Supported Models:** Advanced models like OpenAI’s GPT series are specifically trained to support this natively,.
- **Legacy/Open Source Models:** Smaller or older models (like TinyLlama) may not support this function and require **Output Parsers** (covered in the next lecture),,.

---

## 4. Methods for Defining Schemas

There are three primary ways to define the structure you want the LLM to follow:

### A. TypedDict (Python Native)

- **Definition:** A way to specify which keys and value types should exist in a Python dictionary.
- **Features:** Supports **Annotated** descriptions to guide the LLM, **Optional** fields, and **Literals** (limiting choices to specific values like "Positive" or "Negative"),,.
- **Limitation:** It provides no actual data validation; if the LLM sends a string where an integer was requested, Python will not throw an error,.

### B. Pydantic (Recommended for Python)

- **Definition:** A powerful data validation and parsing library,.
- **Features:**
    - **Validation:** It ensures data is correct and type-safe; it will throw an error if the LLM provides the wrong format,.
    - **Field Function:** Allows you to add constraints (e.g., CGPA must be between 0 and 10) and descriptions,,.
    - **Type Coercion:** Smart enough to convert a string "32" into an integer 32 automatically.
    - **Object Oriented:** Returns a Pydantic object that can be converted to a dictionary or JSON easily,,.

### C. JSON Schema (Cross-Language)

- **Definition:** A universal data format that is language-independent.
- **Use Case:** Best when your project uses multiple languages (e.g., Python backend and JavaScript frontend) and needs a shared schema,.
- **Structure:** Requires defining a `title`, `type` (usually "object"), and `properties` with specific types and descriptions,.

---

## 5. Comparison Table: Choosing a Method

| Feature                  | TypedDict | Pydantic | JSON Schema |
| :----------------------- | :-------- | :------- | :---------- |
| **Basic Structure**      | Yes       | Yes      | Yes         |
| **Type Enforcement**     | Yes       | Yes      | Yes         |
| **Data Validation**      | No        | Yes      | Yes         |
| **Default Values**       | No        | Yes      | No          |
| **Automatic Conversion** | No        | Yes      | No          |
| **Cross-Language**       | No        | No       | Yes         |

---

## 6. Technical Implementation Details

- **Methods of Generation:** The `with_structured_output` function can use two modes: `json_mode` (standard JSON output) or `function_calling` (best for OpenAI and Agents).
- **Rule of Thumb:** Use `function_calling` for OpenAI models and `json_mode` for models like Claude or Gemini,.
