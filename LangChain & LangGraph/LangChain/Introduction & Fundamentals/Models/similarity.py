import sys

sys.path.append(
    "/home/rkroy/Desktop/code/python/LangChain & LangGraph/LangChain/Introduction & Fundamentals/Models"
)

from langchain_core.embeddings import Embeddings
import numpy as np
from numpy.linalg import norm
from simple_embedding import SimpleEmbeddings


class Similarity:
    def __init__(self, embedding_model: Embeddings) -> None:
        self.embedding_model = embedding_model

    def get_similar_document(self, documents: list[str], query: str):
        if isinstance(documents, list) and len(documents) == 0:
            raise ValueError("documents must not be empty")

        if isinstance(query, str) and len(query) == 0:
            raise ValueError("query must not be empty")

        # NOTE: Fine tune documents and query here if needed.

        # You can use the following code if you want to use the embedding model directly
        # query_embedding = self.embedding_model.embed_query(query)
        # documents_embedding = self.embedding_model.embed_documents(documents)
        # But i will use my own way, not very good!!! but who cares :)

        # WARNING: Not recommended
        query_embedding = SimpleEmbeddings(self.embedding_model).embed_query(query)
        documents_embedding = SimpleEmbeddings(self.embedding_model).embed_documents(
            documents
        )

        max_similarity, best_similar_document = float("-inf"), ""
        for i in range(len(documents_embedding)):
            similarity = self.cosine_similarity(query_embedding, documents_embedding[i])
            if similarity > max_similarity:
                max_similarity = similarity
                best_similar_document = documents[i]

        return best_similar_document

    def cosine_similarity(self, vector1, vector2):
        return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
