from abc import ABC, abstractmethod
from typing import Union, List, Optional
from langchain_core.documents import Document

class BaseVectorDB(ABC):
    """base interface for embeddings"""
    
    @abstractmethod
    async def add(
        self,
        embeddings: List[list[float]],
        documents: List[str],
        metadata: Optional[List[dict]] = None
    ) -> List[str]:
        """ 
         Add documents and their embeddings to the database.
        
        Returns:
            List of generated document IDs.
        """
        pass
    
    
    @abstractmethod
    async def search(
        self,
        user_id: str,
        query: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search for similar documents with support for multiple search strategies.
        
        Args:
            query: Raw query string (will be embedded automatically)
            query_vector: Optional pre-computed embedding vector for semantic search
            top_k: Number of results to return
            filter: Legacy filter format (deprecated)
            user_id: User ID for filtering results
        
        Returns:
            List of Documents(page_content, metadata)
        """
        pass
    