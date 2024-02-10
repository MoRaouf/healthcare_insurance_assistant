import uuid
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Qdrant
from langchain import hub
from unstructured.staging.base import elements_from_json
from qdrant_lc import QdrantVectorStore
from pdf_utils import categorize_elements, summarize_table_or_text
from dotenv import load_dotenv

load_dotenv()


# ========================== Retriver ================================

# Categorize PDF elements
raw_pdf_elements = elements_from_json(filename="./data/raw_elements_chunked.json")
text_elements, table_elements = categorize_elements(raw_pdf_elements= raw_pdf_elements)

# Apply to text
texts = [i.text for i in text_elements]
text_summaries = summarize_table_or_text(texts=texts)

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_table_or_text(texts=tables)

# ==========================================

# Retriever
q_client = QdrantVectorStore()
qdrant = Qdrant(client=q_client.client, collection_name="healthcare_demo", embeddings=OpenAIEmbeddings())

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
doc_ids = [str(uuid.uuid4()) for _ in text_elements]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, text_elements)))

# Add tables to MultiVector Retriever
table_ids = [str(uuid.uuid4()) for _ in table_elements]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, table_elements)))


# ============================= Chain ===============================

# Prompt template
template = hub.pull("moraouf/simple_semi_structured_rag_qa:59ce3634")
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# RAG pipeline
semi_structured_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

query = "Can you explain what medical necessity is in relation to coverage?"
# semi_structured_chain.invoke(query)
