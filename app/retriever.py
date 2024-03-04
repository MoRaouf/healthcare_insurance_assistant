import json
import logging
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)



def build_retriever(
        vectorstore_collection_name: str,
)-> MultiVectorRetriever:
    """Builds a MultiVector Retriever with Qdrant as Vector Store & InMemory Doc Store

    Args:
        vectorstore_collection_name (str): Collection name to be created in Qdrant

    Returns:
        MultiVectorRetriever: An instance of MultiVectorRetriever
    """

    with open("./data/processed/pdf_texts.json", "r") as file:
        texts = json.load(file)

    with open("./data/processed/pdf_text_summaries.json", "r") as file:
        text_summaries = json.load(file)

    with open("./data/processed/pdf_tables.json", "r") as file:
        tables = json.load(file)
    
    with open("./data/processed/pdf_table_summaries.json", "r") as file:
        table_summaries = json.load(file)

    logger.info("Loaded PDF elements")

    # ============================ Retriever ================================
    # q_client = QdrantVectorStore()
    # qdrant = Qdrant(client=q_client.client, collection_name="healthcare_demo", embeddings=OpenAIEmbeddings())
    qdrant = Qdrant.construct_instance(
        texts=["test"],
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=vectorstore_collection_name,
        path="./qdrant_db",
    )

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=qdrant,
        docstore=store,
        id_key=id_key,
    )

    # Add texts to MultiVector Retriever
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_text_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]

    retriever.vectorstore.add_documents(
        documents=summary_text_docs,
        # embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        # collection_name="healthcare_demo",
        # path="/qdrant_db",
        )
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    logger.info("Added text documents to retriever")

    # Add tables to MultiVector Retriever
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_table_docs = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_table_docs)
    logger.info("Added table documents to retriever")

    return retriever


if __name__ == "__main__":

    # Check if retriever is built correctly
    collection_name="healthcare_demo"
    retriever = build_retriever(
        vectorstore_collection_name=collection_name,
    )

    print(retriever.vectorstore.similarity_search("provider of insurance"))