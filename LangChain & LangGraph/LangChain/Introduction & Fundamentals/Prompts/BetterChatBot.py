# Better Version of Chat Bot
# To Run --> streamlit run BetterChatBot.py

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("ChatBot")

# Define keys for session state to manage input and output
TEXT_INPUT_KEY = "user_input_text"
LLM_RESPONSE_KEY = "llm_response_content"

# Initialize session state variables if they don't exist
if TEXT_INPUT_KEY not in st.session_state:
    st.session_state[TEXT_INPUT_KEY] = ""
if LLM_RESPONSE_KEY not in st.session_state:
    st.session_state[LLM_RESPONSE_KEY] = ""


# Function to clear the text input and the displayed response
def clear_interaction():
    st.session_state[TEXT_INPUT_KEY] = ""
    st.session_state[LLM_RESPONSE_KEY] = ""


# Use the key argument for st.text_input to bind it to session_state
user_prompt = st.text_input("Enter your Prompt Here", key=TEXT_INPUT_KEY)

# Use columns to place buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Submit"):
        # Check if the text input is not empty before invoking LLM
        if st.session_state[TEXT_INPUT_KEY]:
            with st.spinner("Thinking..."):
                res = llm.invoke(st.session_state[TEXT_INPUT_KEY])
                st.session_state[LLM_RESPONSE_KEY] = res.content
        else:
            st.warning("Please enter a prompt before submitting.")

with col2:
    # Add a clear button that calls clear_interaction function on click
    if st.button("Clear", on_click=clear_interaction):
        pass  # The action is handled by the on_click callback

# Display the LLM response if it exists in session state
if st.session_state[LLM_RESPONSE_KEY]:
    st.write(st.session_state[LLM_RESPONSE_KEY])
