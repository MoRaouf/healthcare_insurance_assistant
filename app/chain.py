from operator import itemgetter

from langchain import hub
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from retriever import build_retriever
from dotenv import load_dotenv

load_dotenv()

# ============================= Build Retriever ===============================
collection_name="healthcare_demo"

retriever = build_retriever(
    vectorstore_collection_name=collection_name,
)

# ============================= Semi-structured Chain ===============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt template
prompt = hub.pull("moraouf/simple_semi_structured_rag_qa_with_chat_history")

# LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# RAG Chain
# rag_chain = (
#     RunnableParallel(
#         {"context": retriever, "question": RunnablePassthrough()} 
#     )
#     | prompt 
#     | model 
#     | StrOutputParser()
# )

# `rag_chain_with_history` manages the invokation, 
# so we removed `{"context": retriever, "question": RunnablePassthrough()}  ` in `rag_chain `
# The output of MultiVectorRetriever is text, so no need to pass its output to `format_docs()`
rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
        )
    | prompt 
    | model 
    | StrOutputParser()
)

# Semi Structured Pipeline with Chat History
rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///rag_chat_history.db"
    ),
    input_messages_key="question",
    history_messages_key="chat_history",
)

# We get the question & context, then assign the output of `rag_chain_with_history` to `answer` key
# `rag_chain_with_history` will run `rag_chain`, which will get `{"context": ..., "question": ...}` from previous step
rag_chain_with_history_and_sources = RunnableParallel(
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
).assign(answer=rag_chain_with_history)


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
# chitchat_chain = (
#     {"question": RunnablePassthrough()} 
#     | prompt 
#     | model 
#     | StrOutputParser()
# )

# `chitchat_chain_with_history` manages the invokation, so we removed `{"question": RunnablePassthrough()} ` in `chitchat_chain `
chitchat_chain = (
    prompt 
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


if __name__ == "__main__":

    # Test rag_chain_with_history
    config = {"configurable": {"session_id": "12345"}}
    output = rag_chain_with_history_and_sources.invoke({"question": "Who is the provider of insurance"}, config=config)
    print(output)

    # Check Retriever output docs & their count
    # chain = RunnablePassthrough.assign(
    #     context=itemgetter("question") | retriever
    #     )
    # output = chain.invoke({"question": "Benefits of insurance"})
    # context = output["context"]
    # print(len(context))
    # print(output)

    # Check similarity of retriever & vectorstore outputs
    # print(retriever.get_relevant_documents("provider of insurance"))
    # print(retriever.vectorstore.similarity_search("provider of insurance"))