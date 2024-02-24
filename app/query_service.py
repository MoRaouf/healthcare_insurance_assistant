from uuid import uuid4
from chain import semi_structured_chain_with_history, chitchat_chain_with_history
from router import route_layer
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class QueryService:

    def _create_session_id(self):
        session_id = uuid4()
        return session_id
    
    def query(
            self,
            question:str,
            session_id:str,
            ):
        session_id = session_id if session_id else self._create_session_id()

        # Configure the session id for the chains
        config = {"configurable": {"session_id": session_id}}

        # Route the input query to the relevant chain
        route = route_layer(question)

        if route.name == "chitchat" or route.name is None:
            logger.info("Selected Route is: ", route.name)
            semi_structured_chain_with_history.invoke({"question": question}, config=config)

        elif route.name == "healthcare_insurance":
            logger.info("Selected Route is: ", route.name)
            chitchat_chain_with_history.invoke({"question": question}, config=config)
    
    def query_route(self, params: Dict):
        return self.query(
            question=params["question"],
            session_id=params["session_id"]
            ) 