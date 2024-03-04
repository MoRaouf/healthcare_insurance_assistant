from typing import Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langserve import CustomUserType, add_routes
from langserve.pydantic_v1 import Field
from query_service import QueryService
from chain import retriever

app = FastAPI(
    title="Healthcare Insurance Assistant Server",
    version="1.0",
    description="A server that runs a chatbot with history",
)

# class InputChat(CustomUserType):
#     """Input for the chat endpoint without displaying Chat History on UI."""

#     question: str = Field(
#         ...,
#         description="The human input question to the chatbot.",
#     )

#     session_id: str = Field(
#         ...,
#         description="The session id for chat history.",
#     )

class InputChat(CustomUserType):
    """Input for the /chat endpoint with displaying Chat History on UI.

    - The order of attributes definition reflects their order of display on UI.
    - The addition of chat_history is necessary to show it on UI
    """

    session_id: str = Field(
        ...,
        description="The session id for chat history.",
    )
    # Specify chat_history as follows to display it on UI
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        examples=[[("human input", "ai response")]],
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )

    question: str = Field(
        ...,
        description="The human input question to the chatbot.",
    )

    # reference_documents: List = Field(
    #     ...,
    #     description="The reference documents from which context is extracted.",
    # )


def _format_to_dict(input: InputChat) -> Dict:
    """Format the input to a dict to be passed to `QueryService.server_query`."""

    # We only need the question & session_id in the dictionary
    return {"question": input.question, "session_id": input.session_id}


# Create Query Service instance
query_service = QueryService()

# InputChat is the input to RunnableLambda, which formats the Pydantic model to dict & pass it to `query_service.server_query`
# Final Chain with Chat History displayed on UI 
# final_chain = RunnableLambda(_format_to_dict).with_types(input_type=InputChat) | RunnableLambda(query_service.server_query)

# Final Chain with Chat History displayed on UI -- & assign the output to `answer` key
final_chain = RunnableParallel(
    {"answer": (
        RunnableLambda(_format_to_dict) | RunnableLambda(query_service.server_query)
        )
    }
).with_types(input_type=InputChat)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(
    app,
    final_chain, 
    config_keys=["configurable"],
    path="/chat"
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)