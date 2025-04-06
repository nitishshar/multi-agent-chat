from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    """
    Class for creating and managing vector stores from document chunks.
    """
    def __init__(self, storage_path='_vector_db'):
        self.storage_path = storage_path
        self.vector_store = None
        self.embedder = OpenAIEmbeddings()

    def build_and_save(self, documents: List[Document]):
        """
        Creates and persists a Chroma vector store from document chunks.
        
        Args:
            documents: List of document chunks to store
        """
        logger.info(f"Creating vector store with {len(documents)} documents in {self.storage_path}")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory=self.storage_path
        )
        # Note: persist() is automatically called when persist_directory is provided
        # in Chroma.from_documents, so we don't need to call it explicitly
        
        logger.info(f"Stored {len(documents)} documents in vector store at {self.storage_path}")
        return self.vector_store
    
    def load(self):
        """
        Loads an existing vector store from storage.
        """
        self.vector_store = Chroma(
            embedding_function=self.embedder,
            persist_directory=self.storage_path
        )
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search on the vector store.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of document chunks
        """
        if not self.vector_store:
            self.load()
        
        results = self.vector_store.similarity_search(query, k=k)
        return results 