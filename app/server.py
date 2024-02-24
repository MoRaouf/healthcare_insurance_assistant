from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from query_service import QueryService

app = FastAPI()

class Input(BaseModel):
    question: str
    session_id: str


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(
    app, 
    RunnableLambda(QueryService.query_route).with_types(input_type=Input), 
    "/rag"
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
