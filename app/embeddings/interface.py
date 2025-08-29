from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np

class BaseEmbeddings(ABC):
    """Abstract base class for asynchronous embedding generators."""

    @abstractmethod
    async def embed_query(
        self,
        text: str | List[str]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings asynchronously for a single string or list of strings.
        
        Returns:
            - List[float]: embedding for a single string
            - List[List[float]]: list of embeddings for multiple strings
        """
        pass

    # @abstractmethod
    # async def get_dimension(self) -> Optional[int]:
    #     """
    #     Return the dimension of the embedding vector.
    #     """
    #     pass