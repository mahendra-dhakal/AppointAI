from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Union,  Generic, TypeVar
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

T = TypeVar("T", bound=BaseChatModel)

class InferenceProvider(ABC, Generic[T]):
    """base interface  for all provider"""


    @abstractmethod
    async def invoke(
        self,
        prompt: str,
        system_prompt: str ="",
        model: str | None =None ,
        **kwargs
        ) -> str :
        """
        Generate a complete response from the LLM
        Args:
            prompt: User query with context
            system_prompt: System instructions for the LLM
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            AsyncGenerator[str, None] : streaming chunk of the chatbot response 
        
        """
        pass
    
    @abstractmethod
    async def stream(self, prompt: Union[str, List[BaseMessage]]):
        """Asynchronously stream the response from the model."""
        pass

 
    
  
