"""Qdrant Vector DB using LangChain"""

import os
from typing import Iterable, List, Optional, Sequence

import qdrant_client
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from qdrant_client.http.models import Filter

# ===================================================================================
# ---------------------------------- Qdrant Class  ----------------------------------
# ===================================================================================


class QdrantVectorStore:
    """Class for Qdrant Vector Store"""

    def __init__(
        self,
        local: str = True,
        url: str = None,
        api_key: str = None,
    ):
        """Instantiate a Client instance"""
        if local:
            os.makedirs("/qdrant_db", exist_ok=True)
            self.client = qdrant_client.QdrantClient(path="/qdrant_db")
        else:
            if url is None or api_key is None:
                raise ValueError(
                    "Please make sure you passed the correct URL & API Key"
                )

            self.client = qdrant_client.QdrantClient(url=url, api_key=api_key)

        self.default_embeddings_model = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    # ===================================================================================

    def list_collections(self):
        # List all existing collections

        return self.client.get_collections()

    # ===================================================================================

    def add_texts(
        self,
        collection_name: str,
        texts: Iterable[str],
        embeddings_model: Embeddings = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        """Add new text data to an existing collection in Qdrant"""

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embeddings_model,
        )

        added_payloads = qdrant.add_texts(
            texts=texts,
            metadatas=metadatas if metadatas is not None else [],
            ids=ids if ids is not None else [],
        )

        print("Added Text Payloads -> Completed")

    # ===================================================================================

    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings_model: Embeddings = None,
    ):
        """Add new documents to an existing collection in Qdrant"""

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embeddings_model,
        )

        added_payloads = qdrant.add_documents(
            documents=documents,
        )

        print("Added Document Payloads -> Completed")

    # ===================================================================================

    def from_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings_model: Embeddings = None,
    ):
        # """Create a Qdrant client, a new collection & insert documents all in one step"""
        """Create a Qdrant client, a new collection & insert documents all in one step

        Args:
            collection_name (str): collection name to store
            documents (List[Document]): list of text dcouments to insert into Qdrant
            embeddings_model (Embeddings, optional): embeddings model to embed input query. Defaults to None.
        """

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant.from_documents(
            documents,
            embeddings_model,
            path="qdrant_db/",
            collection_name=collection_name,
        )

    # ===================================================================================

    def similarity_search(
        self,
        input_query: str,
        collection_name: str,
        embeddings_model: Embeddings = None,
        filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
        top_k=10,
    ):
        """Return docs most similar to query.

        Args:
            collection_name (str): collection name to search
            embeddings_model (Embeddings): embeddings model to embed input query
            input_query (str): Text to look up documents similar to
            filter (Optional[Filter], optional): Filter by metadata. Defaults to None.
            score_threshold (Optional[float], optional): Define a minimal score threshold for the result.
                If defined, less similar results will not be returned. Defaults to None.
            top_k (int, optional): Number of Documents to return. Defaults to 10.

        Returns:
            List[Document]: List of Documents most similar to the query.
        """

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embeddings_model,
        )

        query_results = qdrant.similarity_search(
            query=input_query,
            k=top_k,
            filter=filter,
            score_threshold=score_threshold,
        )

        return query_results

    # ===================================================================================

    def delete_collection(
        self,
        collection_name: str,
    ):
        """Delete collection from Qdrant"""

        self.client.delete_collection(collection_name=collection_name)

    # ===================================================================================

    def qdrant_retriever(
        self,
        collection_name: str,
        embeddings_model: Embeddings = None,
    ):
        """Return Qdrant as a retriever for LangChain

        Args:
            collection_name (str): The collection to search
            embeddings_model (Embeddings): Embeddings model to embed search queries

        Returns:
            Retrieever (VectorStoreRetriever): A qdrant retriever instance
        """

        # Use FastEmbed Default Embedding Model "BAAI/bge-small-en-v1.5"
        if embeddings_model is None:
            embeddings_model = self.default_embeddings_model

        # Qdrant instance from QdrantClient
        qdrant = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embeddings_model,
        )

        return qdrant.as_retriever()
