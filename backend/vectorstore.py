import os
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# set environment variable
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Index name constant
INDEX_NAME = PINECONE_INDEX_NAME

# initialize pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# define embedding models
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# retriever function
def get_retriever():
    """Initialize and returns the Pinecone vectorstore retriever"""
    # ensure the index exists,create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws",region="us-east-1")
        )
        print("Created pinecone index")
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vector_store.as_retriever()

# upload documents to vector store
def add_document(text_content:str):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cant be empty!")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    
    # create langchain document objects from raw text
    documents = text_splitter.create_documents([text_content])

    print("Splitting document into chunks for indexing...")

    # get vector_store instance to add documents
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # add documents to vector_store
    vector_store.add_documents(documents)
    print("Successfully added documents to pinecone vectorstore.")