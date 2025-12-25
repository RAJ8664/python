# Gemini
# This script demonstrates how to use the ChatGoogleGenerativeAI model from langchain_google_genai
# to invoke Google's Gemini language model and generate a response to a prompt.
# It also loads environment variables from a .env file for configuration.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini language model (version 2.5 flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Invoke the model with a prompt to generate Java segment tree code
res = llm.invoke("Can you write me a segment tree code in java")

# Print the generated content from the model's response
print(res.content)
