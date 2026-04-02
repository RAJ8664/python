import streamlit as st
import os
import tempfile
from typing import List
from dotenv import load_dotenv
import PyPDF2
import heapq
import numpy as np
from numpy.linalg import norm
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class Tuple:
    def __init__(
        self, resume_embedding: list[float], idx: int, similarity_score: float
    ) -> None:
        self.resume_embedding = resume_embedding
        self.idx = idx
        self.similarity_score = similarity_score

    def __lt__(self, other):
        return self.similarity_score > other.similarity_score


class TypicalLife:
    def __init__(self, chat_model: BaseChatModel, embedding_model: Embeddings) -> None:
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def get_similarity_score(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Calculate the similarity score between two embeddings.
        Args:
            embedding1 (list[float]): The first embedding.
            embedding2 (list[float]): The second embedding.

        Returns:
            float: The similarity score between the two embeddings.
        """
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def get_top_k_resumes(
        self, job_specification: str, resumes: list[str], k: int
    ) -> list[Tuple]:
        """
        Get the top k resumes based on the job specification.
        Args:
            job_specification (str): The job specification.
            resumes (list[str]): The list of resumes.
            k (int): The number of resumes to return.

        Returns:
            list[str]: The top k resumes.
        """

        job_specification_embedding = self.embedding_model.embed_query(
            job_specification
        )

        resume_embeddings = []
        pq = []
        for i in range(len(resumes)):
            if os.path.isfile(resumes[i]):
                text = self.extract_text_from_pdf(resumes[i])
                if text.strip():
                    embedding = self.embedding_model.embed_query(text)
                    similarity_score = self.get_similarity_score(
                        embedding, job_specification_embedding
                    )
                    resume_embeddings.append(embedding)
                    heapq.heappush(pq, Tuple(embedding, i, similarity_score))

        # Get the top k resumes
        final_resumes = []
        while len(pq) > 0 and k > 0:
            current_resume_info = heapq.heappop(pq)
            final_resumes.append(current_resume_info)
            k -= 1

        return final_resumes


st.set_page_config(
    page_title="Resume Matcher - Find Top Matching Resumes",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "top_k_results" not in st.session_state:
    st.session_state.top_k_results = None
if "resume_texts" not in st.session_state:
    st.session_state.resume_texts = {}


@st.cache_resource
def load_models():
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    embedding_model = OpenAIEmbeddings()
    return TypicalLife(chat_model, embedding_model)


tl = load_models()

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Job Specification")
job_specification = st.sidebar.text_area(
    "Enter or paste the job specification:",
    height=250,
    value="""

**Job Title:** Software Development Engineer

**Location:** Bangalore, India

**Job Type:** Full Time

**Job Description:**  
We are seeking a talented Software Development Engineer to join our dynamic team. You will be responsible for designing, developing, testing, and maintaining high-quality software solutions. The ideal candidate is passionate about technology, thrives in a collaborative environment, and is eager to solve complex problems.

**Key Responsibilities:**
- Design, implement, and maintain scalable software applications.
- Collaborate with cross-functional teams to define, design, and ship new features.
- Write clean, efficient, and well-documented code.
- Participate in code reviews and provide constructive feedback.
- Troubleshoot, debug, and upgrade existing systems.
- Stay up-to-date with emerging technologies and industry trends.

**Requirements:**
- Bachelor's degree in Computer Science, Engineering, or related field (or equivalent experience).
- Proficiency in one or more programming languages (e.g., Python, Java, C++, JavaScript).
- Experience with software development tools and version control systems (e.g., Git).
- Strong problem-solving and analytical skills.
- Excellent communication and teamwork abilities.

**Preferred Qualifications:**
- Experience with cloud platforms (e.g., AWS, Azure, GCP).
- Familiarity with Agile development methodologies.
- Knowledge of databases and RESTful APIs.

**Benefits:**
- Competitive salary and benefits package.
- Flexible working hours and remote work options.
- Opportunities for professional growth and development.""",
)

k = st.sidebar.slider(
    "Select number of top resumes to display:",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Resumes (PDFs)")
    uploaded_files = st.file_uploader(
        "Upload PDF resume files:",
        type="pdf",
        accept_multiple_files=True,
        help="You can select multiple PDF files at once",
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

with col2:
    st.subheader("📊 Uploaded Files")
    if st.session_state.uploaded_files:
        with st.container():
            for i, file in enumerate(st.session_state.uploaded_files, 1):
                st.write(f"**{i}. {file.name}** ({file.size / 1024:.2f} KB)")
    else:
        st.info("No files uploaded yet. Upload PDF files to get started.")

st.divider()
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button(
        "🚀 Find Top Matching Resumes", use_container_width=True, type="primary"
    ):
        if not st.session_state.uploaded_files:
            st.error("❌ Please upload at least one PDF file.")
        elif not job_specification.strip():
            st.error("❌ Please enter a job specification.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                resume_paths = []

                with st.spinner("📥 Processing uploaded files..."):
                    for uploaded_file in st.session_state.uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        resume_paths.append(file_path)

                st.session_state.resume_texts = {}
                with st.spinner("📖 Extracting text from PDFs..."):
                    for i, file_path in enumerate(resume_paths):
                        file_name = st.session_state.uploaded_files[i].name
                        text = tl.extract_text_from_pdf(file_path)
                        st.session_state.resume_texts[i] = {
                            "name": file_name,
                            "text": text,
                            "path": file_path,
                        }

                with st.spinner("🔍 Analyzing resumes with AI..."):
                    results = tl.get_top_k_resumes(job_specification, resume_paths, k)
                    st.session_state.top_k_results = results

if st.session_state.top_k_results is not None:
    st.divider()
    st.subheader("🏆 Top Matching Resumes")

    results = st.session_state.top_k_results

    if results:
        tabs = st.tabs([f"Match #{i + 1}" for i in range(len(results))])

        for tab_idx, x in enumerate(results):
            with tabs[tab_idx]:
                resume_info = st.session_state.resume_texts.get(x.idx)

                if resume_info:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**📄 File Name:** {resume_info['name']}")
                    with col2:
                        similarity_percentage = (
                            (x.similarity_score + 1) / 2 * 100
                        )  # Convert from [-1, 1] to [0, 100]
                        st.metric(
                            "Match Score",
                            f"{similarity_percentage:.2f}%",
                            delta=f"{x.similarity_score:.4f}",
                        )

                    st.divider()

                    # Display resume text
                    with st.expander("📋 View Full Resume Text", expanded=True):
                        st.text_area(
                            "Resume Content:",
                            value=resume_info["text"],
                            height=400,
                            disabled=True,
                            key=f"resume_text_{x.idx}",
                        )
    else:
        st.warning("⚠️ No matching resumes found. Try adjusting your search criteria.")

# Footer
st.divider()
st.markdown(
    """
    """,
    unsafe_allow_html=True,
)
