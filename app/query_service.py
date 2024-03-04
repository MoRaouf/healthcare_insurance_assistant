from uuid import uuid4
from langchain.schema import AIMessage
from chain import rag_chain_with_history, chitchat_chain_with_history
from router import route_layer
from typing import Dict
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class QueryService:

    def _create_session_id(self):
        session_id = uuid4()
        return session_id
    
    def query(
            self,
            question:str,
            session_id:str,
            ) -> AIMessage:
        """Query method for normal usage without LangServe Server

        Args:
            question (str): Input user question
            session_id (str): Session ID of the chat session

        Returns:
            AIMessage: The output of the chain call
        """
        
        # Get or create Session ID
        session_id = session_id if session_id else self._create_session_id()

        # Configure the session id for the chains
        config = {"configurable": {"session_id": session_id}}

        # Route the input query to the relevant chain
        route = route_layer(question)

        if route.name == "chitchat" or route.name is None:
            logger.info(f"Selected Route is: {route.name}")
            output = chitchat_chain_with_history.invoke({"question": question}, config=config)
            return output

        elif route.name == "healthcare_insurance":
            logger.info(f"Selected Route is: {route.name}")
            output = rag_chain_with_history.invoke({"question": question}, config=config)
            return output
    
    def server_query(self, params: Dict):
        """To be used for RunnableLambda in LangServe Server

        Args:
            params (Dict): Dictionary contains the keys of `question` & `session_id`
        """

        return self.query(
            question=params["question"],
            session_id=params["session_id"],
        )

        # # Get question
        # if params.get("question"):
        #     question = params.get("question") 
        # else:
        #     raise ValueError("Please add the input question")
        
        # # Get or create Session ID
        # session_id = params.get("session_id") if params.get("session_id") else self._create_session_id()

        # # Configure the session id for the chains
        # config = {"configurable": {"session_id": session_id}}

        # # Route the input query to the relevant chain
        # route = route_layer(question)

        # if route.name == "chitchat" or route.name is None:
        #     logger.info("Selected Route is: ", route.name)
        #     output = chitchat_chain_with_history.invoke({"question": question}, config=config)
        #     return output

        # elif route.name == "healthcare_insurance":
        #     logger.info("Selected Route is: ", route.name)
        #     output = rag_chain_with_history.invoke({"question": question}, config=config)
        #     return output
    

if __name__ == "__main__":

    input = {"question": "Who is the provider of insurance", "session_id": "12345"}
    query_service = QueryService()
    print(query_service.server_query(input))