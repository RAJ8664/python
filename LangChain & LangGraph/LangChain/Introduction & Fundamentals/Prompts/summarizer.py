from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# NOTE: The extended version of this class can be found inside output parsers section.
class Summarizer:
    """
    A simple class for summarization.
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model

    def simple_summarize(self, content: str | list[str]) -> str:
        if isinstance(content, str) and len(content) == 0:
            raise ValueError("content must not be empty")

        if isinstance(content, list) and len(content) == 0:
            raise ValueError("content must not be empty")

        # NOTE: Fine tune your prompt here according to your need.

        prompt = PromptTemplate(
            template="write me a brief summary of the following text or list of text provided to you \n {content}",
            input_variables=["content"],
        ).format(content=content)

        return str(self.chat_model.invoke(prompt).content)
