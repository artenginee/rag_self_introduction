import streamlit as sl
import subprocess
import sys

# Install PyPDF2 if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
import PyPDF2
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import PyPDF2

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import faiss

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""Hi, I am Gayeon Lee. I am a passionate Data Professional.
Explore my projects and skills to learn more about my potentials."""

@sl.cache_data
def load_data(): 

    file_path = "test.pdf"

    # Open the PDF file
    with open(file_path, "rb") as file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the text
        text = ""

        # Iterate through each page of the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Get the page object
            page = pdf_reader.pages[page_num]

            # Extract text from the page
            text += page.extract_text()

    df1 = pd.DataFrame({'Text': [text]})

    import string
    import re

    def clean(doc):
        # Remove specific characters
        doc = re.sub(r'•', ' ', doc, flags=re.IGNORECASE)
        doc = re.sub(r'\uf0b7', ' ', doc, flags=re.IGNORECASE)
        # Remove blank spaces
        doc = ' '.join(doc.split())

        return doc

    df1['cleaned'] = df1['Text'].apply(clean)

    columns_to_include = ['cleaned']
    data1 = df1[columns_to_include]
    data_list = list(data1.to_records(index=False)) 
    data = [f"{columns_to_include[0]}:{e[0]}\n" for e in data_list]
    
    vectorizer = SentenceTransformer('all-MiniLM-L12-v2')

    if os.path.isfile("vector_cache"):
        with open("vector_cache", 'rb') as f:
            comment_vectors = pickle.load(f)
        f.close()
    else:
        comment_vectors = vectorizer.encode(data)
        with open("vector_cache", 'wb') as f:
            pickle.dump(comment_vectors,f)
    f.close()
    
    dimension = comment_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(comment_vectors)

    return data, vectorizer, index

@sl.cache_resource
def llm_model():
    from langchain_community.llms import HuggingFaceEndpoint

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # auth to huggingface
    HUGGINGFACEHUB_API_TOKEN = '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'
    import os
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=1024,
    temperature=0.1,
    token=HUGGINGFACEHUB_API_TOKEN
    )
    return llm

@sl.cache_data
def inference(q):
    info=""

    if q:
        retrieved_documents = retrieve(q, k=1)
        documents = retrieved_documents
        info = ""
        for document in documents:
            info += document + "\n\n"

    # create template for LLM
    template = """Question: {question}
    Answer: Let's think step by step. The information can be used \n###\n{info}"""
    prompt = PromptTemplate.from_template(template)

    # post q + info to LLM
    llm_chain = LLMChain(prompt=prompt, llm=sl.session_state.llm)
    result = llm_chain.invoke({"question": q, "info": info}, temperature=0.1)

    return result['text']
    
# Retrieve
def retrieve(query, k=1):
    query_vector = sl.session_state.vectorizer.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = sl.session_state.index.search(query_vector, k)
    return [sl.session_state.data[i] for i in indices[0]]

if 'vectorizer' not in sl.session_state:
    print('init')
    sl.session_state.data, sl.session_state.vectorizer, sl.session_state.index = load_data()
    sl.session_state.llm = llm_model()


@sl.experimental_fragment
def chat():
    # Input box for the question
    q = sl.text_input("Your question")
    
    if q=='':
        sl.write('')
    else:
        sl.write(inference(q))


sl.header("Enter any questions you would like to know.")


chat()