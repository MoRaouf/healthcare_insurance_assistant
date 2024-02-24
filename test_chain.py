import uuid
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores.qdrant import Qdrant
from langchain import hub
from unstructured.staging.base import elements_from_json

from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()




# Vector store
# q_client = QdrantVectorStore()
qdrant = Qdrant(
    client=QdrantClient(path="/qdrant_db"), 
    collection_name="healthcare_demo", 
    embeddings=OpenAIEmbeddings(model= "text-embedding-3-small")
    )

qdrant.from_texts(
    texts="Hello World",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    # collection_name="healthcare_demo",
    # path="/qdrant_db",
    )