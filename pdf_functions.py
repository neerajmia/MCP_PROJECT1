
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from InstructorEmbedding import INSTRUCTOR

# get the raw text from pdf

def get_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
# Tokenize and preprocess the  the pdf text into words

def get_text_tokens(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# generate embeddings using hugguging face


def generate_embedding(text: str,embedding_url: str, hf_token: str ) -> list[float]:
 
  response = requests.post(embedding_url, headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()
