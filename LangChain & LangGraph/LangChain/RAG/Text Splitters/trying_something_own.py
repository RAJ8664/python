"""
we have text splitter (Length based, Text Structure based, Document Structure based)

Consider a text:

The Amazon rainforest, often referred to as the "lungs of the Earth," plays a crucial role in regulating the planet's climate and supports an extraordinary variety of wildlife, many of which are found nowhere else.
Quantum computing, by leveraging the principles of quantum mechanics, promises to solve complex problems far beyond the reach of classical computers, potentially transforming industries such as cryptography, pharmaceuticals, and logistics.

Engaging in regular physical activity not only strengthens the cardiovascular system and muscles but also has profound benefits for mental well-being, including reducing stress, improving mood, and enhancing cognitive function.

if you think properly none of the above approache is good for this scenario, since the each sentences is talking about diff topc.

How to solve this: ?

trying something out of own!!!

"""

from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class SemanticTextSplitter:
    def __init__(self, embedding_model: Embeddings) -> None:
        self.embedding_model = embedding_model
        self.threshold = 0

    def split_into_sentences(self, text: str) -> list[str]:
        current = ""
        sentences: list[str] = []
        for i in range(len(text)):
            if text[i] == ".":
                current += text[i]
                sentences.append(current)
                current = ""
            else:
                current += text[i]

        if len(current):
            sentences.append(current)
            current = ""

        return sentences

    def compute_dynamic_threshold(self, embeddings: list[list[float]]) -> float:
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
        mean = np.mean(similarities)
        std = np.std(similarities)
        return float(0.5 * std + mean)  # You may tweak here

    def can_merge_sentences(
        self,
        sentence1: list[float],
        sentence2: list[float],
        embeddings: list[list[float]],
    ):
        # Uses a dynamic threshold computed from all embeddings
        vec1 = np.array(sentence1)
        vec2 = np.array(sentence2)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cos_sim > self.threshold

    def semantic_split(self, input_data: str | list[Document]) -> list[Document]:
        """
        Splits input data (either a string or a list of Documents) into semantically-merged Documents.
        """
        if isinstance(input_data, str):
            # Input is a string: split into sentences and assign default metadata
            sentences: list[str] = self.split_into_sentences(input_data)
        elif isinstance(input_data, list):
            # Input is a list of Documents: extract sentences and metadata
            sentences: list[str] = []
            for doc in input_data:
                current_split = self.split_into_sentences(doc.page_content)
                for sentence in current_split:
                    sentences.append(sentence)
        else:
            raise ValueError("Input must be a string or a list of Document objects.")

        # Generate embeddings for each sentence
        embeddings: list[list[float]] = []
        for sentence in sentences:
            current_embedding = self.embedding_model.embed_query(sentence)
            embeddings.append(current_embedding)

        # Compute dynamic threshold
        self.threshold = self.compute_dynamic_threshold(embeddings)

        # Merge sentences
        merge_range = []
        left, right, length = 0, 0, len(embeddings)
        while left < length:
            right = left + 1
            while right < length and self.can_merge_sentences(
                embeddings[right - 1], embeddings[right], embeddings
            ):
                right += 1
            merge_range.append((left, right - 1))
            left = right

        # Merge the sentences/documents if similar (Get it from merge_range list)
        splitted_docs: list[Document] = []
        for low, high in merge_range:
            merged_text = " ".join(sentences[low : high + 1]).strip()
            if len(merged_text) == 0:
                continue
            splitted_docs.append(
                Document(page_content=merged_text, metadata={})  # TODO: add metadata
            )

        return splitted_docs


if __name__ == "__main__":
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()

    splitter = SemanticTextSplitter(embeddings)

    # NOTE: works both way

    # text = """The Amazon rainforest, often referred to as the "lungs of the Earth," plays a crucial role in regulating the planet's climate and supports an extraordinary variety of wildlife, many of which are found nowhere else. Quantum computing, by leveraging the principles of quantum mechanics, promises to solve complex problems far beyond the reach of classical computers, potentially transforming industries such as cryptography, pharmaceuticals, and logistics. Quantum computing is a hard topic to learn when learning initially, but you will enjoy a lot. Engaging in regular physical activity not only strengthens the cardiovascular system and muscles but also has profound benefits for mental well-being, including reducing stress, improving mood, and enhancing cognitive function."""
    docs = TextLoader("temp.txt").load()
    splitted_docs = splitter.semantic_split(docs)

    print(len(splitted_docs))
    print(splitted_docs)
