# DirectoryLoader
"""
DirectoryLoader is a document loader in LangChain used to load multiple documents from a directory(folder) or files.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

loader = DirectoryLoader(
    path="/home/rkroy/Downloads", glob="*.pdf", loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))


# Why So slow ?

"""
| Aspect                        | `load()`                                    | `lazy_load()`                              |
| ----------------------------- | ------------------------------------------- | ------------------------------------------ |
| **What it does**              | Loads **all documents at once** into memory | Loads documents **one-by-one (on demand)** |
| **Return type**               | `List[Document]`                            | `Iterator[Document]`                       |
| **Memory usage**              | ❌ Higher (everything stored in RAM)        | ✅ Lower (streamed, memory-efficient)      |
| **Performance on large data** | ❌ Can be slow / crash on large files       | ✅ Scales well for large datasets          |
| **Execution style**           | Eager loading                               | Lazy / streaming                           |
| **Use case**                  | Small to medium datasets                    | Large files, logs, PDFs, web pages         |
| **Processing style**          | Batch processing                            | Streaming / pipeline processing            |
| **Example usage**             | `docs = loader.load()`                      | `for doc in loader.lazy_load():`           |
| **Best for**                  | Quick prototyping, small inputs             | Production pipelines, embeddings, chunking |

"""
