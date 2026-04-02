from langchain_core.embeddings import Embeddings


class SimpleEmbeddings:
    """
    A simple class for embedding queries and documents.
    """

    def __init__(self, embedding_model: Embeddings) -> None:
        self.embedding_model = embedding_model

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string using the provided embedding model.

        Args:
            query (str): The query string to embed.

        Returns:
            list[float]: The embedded query vector.
        """

        if isinstance(query, str) and len(query) == 0:
            raise ValueError("query must not be empty")

        # NOTE: Fine tune prompt here if needed.

        return self.embedding_model.embed_query(query)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed a list of documents using the provided embedding model.

        Args:
            documents (list[str]): The list of documents to embed.

        Returns:
            list[list[float]]: The embedded document vectors.
        """

        if isinstance(documents, list) and len(documents) == 0:
            raise ValueError("documents must not be empty")

        # NOTE: Fine tune documents here if needed.

        return self.embedding_model.embed_documents(documents)
