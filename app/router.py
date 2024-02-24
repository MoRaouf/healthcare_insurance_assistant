from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer
from dotenv import load_dotenv

load_dotenv()

chitchat = Route(
    name="chitchat",
    utterances=[
        "how are you?",
        "how are things going?",
        "what is your name?",
        "how's the weather today?",
    ],
)

healthcare_insurance = Route(
    name="healthcare_insurance",
    utterances=[
        "what are the available benefits of the travel healthcare insurance?",
        "what does the travel healthcare insurance policy cover?",
        "what is the criteria to avail the benefits of the travel healthcare insurance?",
        "who is the provider of the travel healthcare insurance?",
        "what are the terms applicable in case of emergency in the policy",
    ],
)

routes = [healthcare_insurance, chitchat]
encoder = OpenAIEncoder()

route_layer = RouteLayer(encoder=encoder, routes=routes)



if __name__ == "__main__":

    rt = route_layer("benefits of insurance")

    if rt.name == "chitchat":
        print("Selected Route is: ", rt.name)
    elif rt.name == "healthcare_insurance":
        print("Selected Route is: ", rt.name)
    else:
        print("Selected Route is: ", rt.name)