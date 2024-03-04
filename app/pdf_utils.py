import json
import logging
import os
from collections import Counter
from typing import List

from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_from_json, elements_to_json

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
 
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


def process_pdf(filename: str):
    """Categorize, summarize & save PDF elements.

    Args:
        filename (str): PDF file to be processed
    """

    directory = "./data/processed/"
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Categorize PDF elements
    raw_pdf_elements = elements_from_json(filename=filename)
    text_elements, table_elements = categorize_elements(raw_pdf_elements= raw_pdf_elements)

        # Apply to text
    # texts = [i.text for i in text_elements]
    texts = text_elements
    text_summaries = summarize_table_or_text(texts=texts)

    with open("./data/processed/pdf_texts.json", "w") as file:
        json.dump(texts, file)

    with open("./data/processed/pdf_text_summaries.json", "w") as file:
        json.dump(text_summaries, file)

    # Apply to tables
    tables = [i.text for i in table_elements]
    table_summaries = summarize_table_or_text(texts=tables)

    with open("./data/processed/pdf_tables.json", "w") as file:
        json.dump(tables, file)

    with open("./data/processed/pdf_table_summaries.json", "w") as file:
        json.dump(table_summaries, file)

    logger.info("Categorized & processed PDF elements")

if __name__ == "__main__":

    # Chunk & process the PDF -- saves the result automatically
    pdf_file_path = "./data/travel_health_insurance_policy.pdf"
    raw_elements_chunked="./data/raw_elements_chunked.json"
    
    chunk_pdf(pdf_file_path=pdf_file_path)
    process_pdf(filename=raw_elements_chunked)