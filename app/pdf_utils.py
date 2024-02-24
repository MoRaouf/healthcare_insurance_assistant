from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json, elements_from_json
from unstructured.documents.elements import Element
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from collections import Counter

 
pdf_file_path = "./data/travel_health_insurance_policy.pdf"

def chunk_pdf(pdf_file_path: str):
    #Get elements
    raw_pdf_elements = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        infer_table_structure=True,   # this will enable strategy="hi_res"
        extract_images_in_pdf=False,
        # Post processing to aggregate text once we have the title 
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        max_characters=4000,             # Require maximum chunk size of 4000 chars
        new_after_n_chars=3800,          # Attempt to create a new chunk at 3800 chars
        combine_text_under_n_chars=2000, # Attempt to keep chunks > 2000 chars
    )

    elements_to_json(elements=raw_pdf_elements, filename="./data/raw_elements_chunked.json")

def count_elements(pdf_file_path: str):

    raw_pdf_elements = elements_from_json(filename=pdf_file_path)
    print(len(raw_pdf_elements))
    unique_elements = [str(type(el)) for el in raw_pdf_elements]
    counts = Counter(unique_elements)
    # print(counts, "\n\n")
    return counts


def categorize_elements(raw_pdf_elements: List[Element]):
    # Categorize by type
    text_elements = []
    table_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Text" in str(type(element)):
            text_elements.append(str(element))
        # elif "unstructured.documents.elements.Image" in str(type(element)):
        #     image_elements.append(element)
        elif "unstructured.documents.elements.Table" in str(type(element)):
            table_elements.append(element)
        
    return text_elements, table_elements


def summarize_table_or_text(texts: List[str]):
    prompt_template = hub.pull("moraouf/summarize_table_or_text")

    prompt = ChatPromptTemplate.from_template(prompt_template.template)
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    
    return summaries




# if __name__ == "__main__":

    # chunk_pdf(pdf_file_path=pdf_file_path)
    
    