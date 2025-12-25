# Example of dynamic prompt in langchain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("Research Paper Summarizer")

col1, col2, col3 = st.columns(3)

paper_title, paper_style, paper_length = "", "", ""

with col1:
    paper_title = st.text_input("Enter the title of the paper")

with col2:
    paper_style = st.selectbox(
        "Select the style of the paper", ["Scientific", "Business", "Technical"]
    )

with col3:
    paper_length = st.selectbox(
        "Select the length of the paper", ["Short", "Medium", "Long"]
    )

# After taking these input from the user, let's create a dynamic template
template = "I want you to create a summary paper for the paper : {paper_title} in the style of {paper_style} with a length varying betweeen {paper_length} sentence"

# Note : --> The better the prompt is, the better the output, so try to create a good prompt

# Let's create a prompt out of the template
prompt = PromptTemplate.from_template(template)

# Now let's format the prompt with the input values
formatted_prompt = prompt.format(
    paper_title=paper_title, paper_style=paper_style, paper_length=paper_length
)


# Not checking if the input is empty, assuming that the user will enter the values
if st.button("Submit"):
    res = llm.invoke(formatted_prompt)
    st.write(res.content)
