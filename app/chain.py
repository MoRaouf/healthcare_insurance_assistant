from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from qdrant_lc import QdrantVectorStore

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")

# Retriever
client = QdrantVectorStore()
client.add_documents(collection_name="healthcrae_demo",
                     documents="")
retriever = client.qdrant_retriever("healthcare_demo")

# RAG pipeline
semi_structured_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

query = "Can you explain what medical necessity is in relation to coverage?"
# semi_structured_chain.invoke(query)
