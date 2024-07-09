import os

from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def inject_metadata(doc):
    content = doc.page_content

    try:
        source_file = os.path.basename(doc.metadata['source'])
    except:
        source_file = doc.metadata['source']

    return f"Name of the file for this content: `{source_file}` Content: `{content}`"
    
def create_vector_store_index(qdrant_url, collection_name, file_path, embedding):

    if not file_path:
        return "Please try again"
        
    file_path_split = file_path.split(".")
    file_type = file_path_split[-1].rstrip('/')

    if file_type == 'csv':
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
    
    elif file_type == 'pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 128,)

        documents = text_splitter.split_documents(pages)
    else:
        loader = TextLoader(file_path=file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 128,)

        documents = text_splitter.split_documents(pages)

    enriched_documents = [Document(inject_metadata(doc), metadata=doc.metadata) for doc in documents]
    
    qdrant_client = QdrantClient(
        location=qdrant_url)
    
    if collection_name in [collection.name for collection in qdrant_client.get_collections().collections]:
        existing_collection_client = Qdrant(qdrant_client, collection_name, embedding)
        existing_collection_client.add_documents(
            enriched_documents
        )
    else:
        Qdrant.from_documents(
            enriched_documents,
            embedding,
            url=qdrant_url,
            collection_name=collection_name
        )

    return "Vector store index is created."