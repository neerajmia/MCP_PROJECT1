import os
import streamlit as st
from dotenv import load_dotenv
import requests
import time
import pymongo
from pymongo.collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding_urls import embedding_model
#from utils.pdf_functions import get_pdf, get_text_tokens
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

# ✅ Load environment variables
load_dotenv()

# ✅ Load Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROG_API_KEY1")

# ✅ Initialize the MCP server
mcp = FastMCP("ragserver")

# ✅ ---------------------- MongoDB Connection ----------------------

def connect_mongodb() -> Collection:
    """
    Connect to MongoDB and return the collection object.
    """
    mongo_uri = os.getenv("MONGODB_URI")
    mongo_db = os.getenv("MONGODB_DB", "pdf_chat1")
    mongo_collection = os.getenv("MONGODB_COLLECTION", "embeddings")

    client = pymongo.MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    # Create indexes for efficient retrieval
    collection.create_index([("file", pymongo.ASCENDING)])        # Index on file name
    collection.create_index([("embedding", pymongo.TEXT)])        # Text index on embeddings

    return collection

# Initialize MongoDB collection
collection = connect_mongodb()

# ✅ ---------------------- Embedding Functions ----------------------
def generate_embedding(text: str):
    """
    Generate embedding for given text using your embedding model.
    """
    return embedding_model.encode([text]).tolist()

def retrieve_top_chunks_across_files(query_embedding, top_k=3):
    """
    Retrieve top-k most similar text chunks based on cosine similarity.
    """
    results = list(collection.find({}))
    if not results:
        return []

    embeddings = [doc["vector"] for doc in results]
    texts = [doc["text"] for doc in results]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_indices]

# ✅ ---------------------- Answer Generation Tool ----------------------

@mcp.tool()
async def generate_answer_across_files_tool(user_question: str) -> str:
    """
    RAG Answer Generation: Finds top context chunks and generates an answer.
    """
    try:
        query_embedding = generate_embedding(user_question)[0]
        top_chunks = retrieve_top_chunks_across_files(query_embedding)
        context = "\n".join(top_chunks)

        prompt_template = PromptTemplate.from_template(
            "You are an expert. Answer the question based on the context below and compliment it with some already learned examples:\n\n"
            "Context: {context}\n\n"
            "Question: {question}"
        )

        llm = ChatGroq(model="llama3-70b-8192")
        output_parser = StrOutputParser()
        chain = prompt_template | llm | output_parser
        response = chain.invoke({"context": context, "question": user_question})

        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ ---------------------- Streamlit UI ----------------------

user_question = st.text_input("Ask a question about your documents:", key="main_question")

if user_question:
    try:
        # Streamlit can't directly run async, so wrap it synchronously
        import asyncio
        answer = asyncio.run(generate_answer_across_files_tool(user_question))
        st.write("### Answer:")
        st.write(answer)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ✅ ---------------------- MCP Run ----------------------
if __name__ == "__main__":
    mcp.run()