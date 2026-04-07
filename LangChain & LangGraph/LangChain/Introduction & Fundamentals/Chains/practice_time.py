from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


class QUIZ(BaseModel):
    question: list[str] = Field(description = "List of all the questions of the quiz")
    options: list[str] = Field(description = "List of answer of a particular question")
    correct_option: str = Field(description="Correct answer of this current question")


class NOTES(BaseModel):
    topic: list[str] = Field(description="Name of the topic")
    descripttion: list[str] = Field(description="Description or explanation of current topic")

class PracticeTime:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model


    def generate_practice(self, topic: str, additional_description: list[str] | None):
        # TODO:complete this 



