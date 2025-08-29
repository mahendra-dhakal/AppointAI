from abc import ABC, abstractmethod
from typing import Optional, List
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """Abstract base class for document retrieval systems."""

    @abstractmethod
    async def query(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Document]:
        """Query the retriever asynchronously and return a list of documents."""
        pass
    
    