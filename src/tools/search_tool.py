from crewai.tools import BaseTool
from typing import Any, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import Field
import time

# Simple cache implementation
class SimpleCache:
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
        return None
    
    def set(self, key, value):
        self.cache[key] = (time.time(), value)

# Global cache instance
_search_cache = SimpleCache(ttl=300)

class VectorStoreSearchTool(BaseTool):
    name: str = "Vector Store Search"
    description: str = "Search for relevant information in the knowledge base"
    storage_path: str = Field(default="_vector_db")
    embedder: Any = Field(default_factory=OpenAIEmbeddings)

    def __init__(self, storage_path: str = "_vector_db", **kwargs):
        # First, call the parent initializer with all provided keyword arguments.
        super().__init__(**kwargs)
        # Now, set or override additional attributes.
        self.embedder = OpenAIEmbeddings()
        self.storage_path = storage_path

    def _search_vector_store(self, query: str, k: int = 4):
        # Check cache first
        cache_key = f"{query}:{k}"
        cached_result = _search_cache.get(cache_key)
        if cached_result:
            return cached_result
            
        # If not in cache, perform search
        vector_store = Chroma(
            persist_directory=self.storage_path,
            embedding_function=self.embedder
        )
        results = vector_store.similarity_search(query, k=k)
        
        # Save to cache
        _search_cache.set(cache_key, results)
        return results

    def _run(self, query: str, k: Optional[int] = 4) -> str:
        """Execute the tool's search functionality."""
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

    async def _arun(self, query: str, k: Optional[int] = 4) -> str:
        """Async version of the tool's search functionality."""
        # For now, just call the synchronous version
        return self._run(query, k=k)

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
    def _run(self, query: str) -> str:
        return self.run(query)

    async def _arun(self, query) -> str:
        return self.run(query)