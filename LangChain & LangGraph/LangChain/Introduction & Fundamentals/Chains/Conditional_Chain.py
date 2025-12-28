from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

"""

Consider a scenario 
1. User submits a feedback on a product.
2. Analyze the sentiments of the feedback(positive, negative, neutral).
3. If the feedback is positive, then generate a positive review(Reply).
4. If the feedback is negative, then generate a negative review(Reply).
5. If the feedback is neutral, then generate a neutral review(Reply).

┌───────────────────────────┐
│        User Input         │
│  (Product Feedback Text)  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Input Preprocessing      │
│  - Clean text             │
│  - Normalize input        │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Sentiment Analysis Chain │
│  (LLM / Classifier Chain) │
│                           │
│  Output:                  │
│  - "positive"             │
│  - "negative"             │
│  - "neutral"              │
└─────────────┬─────────────┘
              │
              ▼
        ┌───────────────┐
        │ Conditional   │
        │ Router Chain  │
        └───────┬───────┘
                │
   ┌────────────┼────────────┐
   │            │            │
   ▼            ▼            ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Positive  │ │ Negative  │ │ Neutral   │
│ Reply     │ │ Reply     │ │ Reply     │
│ Chain     │ │ Chain     │ │ Chain     │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      │             │             │
      ▼             ▼             ▼
┌────────────────────────────────────┐
│   Generated Review / Response      │
│  (Tone matches sentiment)          │
└─────────────┬──────────────────────┘
              │
              ▼
┌───────────────────────────┐
│        Final Output       │
│  (Sentiment + Reply)      │
└───────────────────────────┘

"""


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the feedback"
    )


# Parsers
parser = StrOutputParser()
sentiment_parser = PydanticOutputParser(pydantic_object=Feedback)

# Sentiment
sentiment_template = PromptTemplate(
    template="Classify the sentiment of the following feedback:\n {feedback} \n {format}",
    input_variables=["feedback"],
    partial_variables={"format": sentiment_parser.get_format_instructions()},
)
feedback_chain = sentiment_template | llm | sentiment_parser

# Positive
positive_template = PromptTemplate(
    template="Generate a positive review to the customer for the following feedback:\n {feedback}",
    input_variables=["feedback"],
)
positive_chain = positive_template | llm | parser

# Negative
negative_template = PromptTemplate(
    template="Generate a negative review to the customer for the following feedback:\n {feedback}",
    input_variables=["feedback"],
)
negative_chain = negative_template | llm | parser

# Neutral
neutral_template = PromptTemplate(
    template="Generate a neutral review to the customer for the following feedback:\n {feedback}",
    input_variables=["feedback"],
)
neutral_chain = neutral_template | llm | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_chain),
    (lambda x: x.sentiment == "negative", negative_chain),
    (lambda x: x.sentiment == "neutral", neutral_chain),
    RunnableLambda(lambda x: "Could not find sentiment"),
)

chain = feedback_chain | branch_chain
res = chain.invoke({"feedback": "This is a bad product"})
print(res)

print(chain.get_graph().print_ascii())
