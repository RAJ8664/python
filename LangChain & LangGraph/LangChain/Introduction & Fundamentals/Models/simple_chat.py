from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import StrOutputParser


class SimpleChat:
    """
    A simple chat class that uses a BaseChatModel to get the response of a prompt.
    This also works with HuggingFace models.
    chat_model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="mistralai/Mistral-7B-Instruct-v0.2"))
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model

    def get_formatted_response(self, prompt: str | list[str]) -> str:
        """
         Get the response of the prompt in string format

        Args:
            chat_model (BaseChatModel): The chat model to use for the response

        Returns:
            Response of the prompt in string format
        """

        if isinstance(prompt, str) and len(prompt) == 0:
            raise ValueError("prompt must not be empty")

        if isinstance(prompt, list) and len(prompt) == 0:
            raise ValueError("prompt must not be empty")

        # NOTE: Fine tune prompt here if needed.

        parser = StrOutputParser()

        chain = self.chat_model | parser

        res = chain.invoke(prompt)

        return res

    def get_complete_response(self, prompt: str | list[str]) -> AIMessage:
        """
         Get the full response of the prompt sent by the chat model

        Args:
            chat_model (BaseChatModel): The chat model to use for the response

        Returns:
            Complete response of the prompt sent by the chat model
        """

        if isinstance(prompt, str) and len(prompt) == 0:
            raise ValueError("prompt must not be empty")

        if isinstance(prompt, list) and len(prompt) == 0:
            raise ValueError("prompt must not be empty")

        # NOTE: Fine tune prompt here if needed.

        return self.chat_model.invoke(prompt)
