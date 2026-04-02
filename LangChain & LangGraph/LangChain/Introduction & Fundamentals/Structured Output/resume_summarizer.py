from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from PyPDF2 import PdfReader
from typing import Optional


class Resume(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    candidate_education: list[str] = Field(
        description="Education List of the candidate"
    )
    candidate_experience: Optional[list[str]] = Field(
        description="List of experiences the candidate have in their resume"
    )
    candidate_projects: Optional[list[str]] = Field(
        description="List of projects the candidate have in their resume"
    )
    candidate_skills: Optional[list[str]] = Field(
        description="Skills List of the candidate"
    )
    candidate_achievements: Optional[list[str]] = Field(
        description="Achievements List of the candidate"
    )
    summary: str = Field(description="Summary of the candidate in brief")


class ResumeSummary:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model

    def summarize_resume(self, resume_pdf_path) -> dict:
        # Extract text from PDF
        reader = PdfReader(resume_pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        parser = PydanticOutputParser(pydantic_object=Resume)
        prompt = PromptTemplate(
            template="Based on the resume content \n {text}, tell me everything about candidate in the format: \n {format}",
            input_variables=["text"],
            partial_variables={"format": parser.get_format_instructions()},
        )

        chain = prompt | self.chat_model | parser

        res = dict(chain.invoke({"text": text}))

        return res
