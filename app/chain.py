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
from qdrant_lc import QdrantVectorStore
from pdf_utils import categorize_elements, summarize_table_or_text
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


# ========================== Retriver ================================

# Categorize PDF elements
raw_pdf_elements = elements_from_json(filename="./data/raw_elements_chunked.json")
text_elements, table_elements = categorize_elements(raw_pdf_elements= raw_pdf_elements)

# Apply to text
# texts = [i.text for i in text_elements]
texts = text_elements
text_summaries = summarize_table_or_text(texts=texts)

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_table_or_text(texts=tables)

# ==========================================

# Vector store
# q_client = QdrantVectorStore()
qdrant = Qdrant(client=QdrantClient(path="/qdrant_db"), collection_name="healthcare_demo", embeddings=OpenAIEmbeddings(model= "text-embedding-3-small"))

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# Retriever
retriever = MultiVectorRetriever(
    vectorstore=qdrant,
    docstore=store,
    id_key=id_key,
)

# ==========================================

# Add texts to MultiVector Retriever
doc_ids = [str(uuid.uuid4()) for _ in text_elements]
summary_text_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]

retriever.vectorstore.from_documents(
    documents=summary_text_docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    # collection_name="healthcare_demo",
    # path="/qdrant_db",
    )
retriever.docstore.mset(list(zip(doc_ids, text_elements)))

# Add tables to MultiVector Retriever
table_ids = [str(uuid.uuid4()) for _ in table_elements]
summary_table_docs = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_table_docs)
retriever.docstore.mset(list(zip(table_ids, table_elements)))

# ============================= Semi-structured Chain ===============================

# Prompt template
prompt_template = hub.pull("moraouf/simple_semi_structured_rag_qa_with_chat_history")
prompt = ChatPromptTemplate.from_template(prompt_template.template)

# LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Semi Structured Chain
semi_structured_chain = (
    RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()} 
    )
    | prompt 
    | model 
    | StrOutputParser()
)

# Semi Structured Pipeline with Chat History
semi_structured_chain_with_history = RunnableWithMessageHistory(
    semi_structured_chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///rag_chat_history.db"
    ),
    input_messages_key="question",
    history_messages_key="chat_history",
)

# ============================= ChitChat Chain ===============================
# Prompt template
prompt_template = """You are a helpful assistant. Answer the user question based on the following chat history.

### Chat History:
{chat_history}

### Question: 
{question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# ChitChat Chain
chitchat_chain = (
    {"question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

# ChitChat Pipeline with Chat History
chitchat_chain_with_history = RunnableWithMessageHistory(
    chitchat_chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///rag_chat_history.db"
    ),
    input_messages_key="question",
    history_messages_key="chat_history",
)


# if __name__ == "__main__":

#     print(type(raw_pdf_elements))
#     print(raw_pdf_elements)
    # print(text_elements[0])
    # print(table_elements[0])

    # query = "Can you explain what medical necessity is in relation to coverage?"
    # output = semi_structured_chain.invoke(query)
    # print(output)