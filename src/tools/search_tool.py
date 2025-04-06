from crewai.tools import BaseTool
from typing import Type
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

class VectorStoreSearchTool(BaseTool):
    """
    Tool for searching the vector store for relevant information.
    """
    name: str = "Vector Store Search"
    description: str = "Search for relevant information in the knowledge base"
    
    def __init__(self, storage_path='_vector_db'):
        """
        Initialize the search tool with the vector store path.
        
        Args:
            storage_path: Path to the vector store
        """
        self.storage_path = storage_path
        self.embedder = OpenAIEmbeddings()
        super().__init__()
    
    @st.cache_data(ttl=300)  # Cache search results for 5 minutes
    def _search_vector_store(self, query: str, k: int = 4):
        """
        Cached function to search the vector store.
        
        Args:
            query: The search query
            k: Number of results to return
        """
        vector_store = Chroma(
            persist_directory=self.storage_path,
            embedding_function=self.embedder
        )
        results = vector_store.similarity_search(query, k=k)
        return results
    
    def run(self, query: str, k: int = 4) -> str:
        """
        Run the search tool.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            String with the search results
        """
        try:
            results = self._search_vector_store(query, k)
            if not results:
                return "No relevant information found in the knowledge base."
            
            output = f"Found {len(results)} relevant document(s):\n\n"
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                source = metadata.get('source_filename', 'Unknown source')
                headers = ""
                
                for level in range(1, 5):
                    level_key = f"Level {level}"
                    if level_key in metadata:
                        headers += f"{metadata[level_key]} > "
                
                if headers:
                    headers = headers.rstrip(" > ")
                
                content = doc.page_content.strip()
                output += f"Document {i}:\n"
                output += f"Source: {source}\n"
                output += f"Context: {headers}\n"
                output += f"Content: {content}\n\n"
                
            return output
        except Exception as e:
            return f"Error searching the knowledge base: {str(e)}"


class AskForClarificationsTool(BaseTool):
    """
    Tool for asking clarifying questions to the user.
    Currently just a placeholder as Streamlit-based UI will handle this differently.
    """
    name: str = "Ask for clarifications"
    description: str = "Ask the user for clarifications when the request is unclear"
    
    def run(self, question: str) -> str:
        """
        This method would typically ask the user for clarification,
        but in a Streamlit app this will be handled via the UI.
        
        Args:
            question: The clarification question to ask
            
        Returns:
            String with placeholder response
        """
        return f"[In a Streamlit app, the clarification '{question}' would be handled via the UI]" 