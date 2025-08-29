from typing import AsyncGenerator, List, Union
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from loguru import logger

from app.config.settings import Settings
from app.inference.interface import InferenceProvider


class GeminiLLMManager(InferenceProvider[BaseChatModel]):
    def __init__(self):
        self._config = Settings
        self._model_id: str = self._config.GEMINI_MODEL_ID or "gemini-1.5-flash"
        self._temperature: float =0.3
        self._model: ChatGoogleGenerativeAI | None = None
        self._initialized: bool = False


    async def initialize(self) -> None:
        """Asynchronous initialization."""
        try:
            self._model = ChatGoogleGenerativeAI(
                model=self._model_id,
                api_key=self._config.GEMINI_API_KEY,
                disable_streaming=False,
                temperature=self._temperature
            )
            self._initialized = True
            logger.info(f"[Gemini] Successfully initialized model: {self._model_id}")
        except Exception as e:
            logger.error(f"[Gemini] Failed to initialize: {e}")
            raise


    

    async def invoke(self, prompt: Union[str, List[BaseMessage]]) -> str:
        """Invoke the model asynchronously and return the response"""
        
        if not self._model:
            raise RuntimeError("Gemini model not initialized")
        try:
            response = await self._model.ainvoke(prompt)
            if isinstance(response.content, str):
                return response.content
            else:
                raise ValueError("Gemini returned non-string content")
        except Exception as e:
            logger.error(f"[Gemini] Invoke error: {e}")
            raise



    async def stream(self, prompt: Union[str, List[BaseMessage]]) -> AsyncGenerator[str, None]:
        """stream the response asynchronously as chunks"""
        
        if not self._model:
            raise RuntimeError("Gemini model not initialized")
        
        buffer = ""

        try:
            async for chunk in self._model.astream(prompt):
                
                if isinstance(chunk.content, str):
                    buffer += chunk.content
                    
                    # This pattern splits on '.', '!', '?' followed by space or end of string.
                    sentences = re.split(r'(?<=[.!?])\s+', buffer)   #split buffer into individual sentence
                    
                    # If the last sentence doesn't end with punctuation, it's incomplete
                    if not re.search(r'[.!?]$', buffer):
                        # Keep the last fragment in the buffer
                        buffer = sentences.pop() if sentences else ""
                    else:
                        # All are full sentences; clear buffer
                        buffer=""
                    
                    for sentence in sentences:
                        cleaned = sentence.strip()
                        if cleaned:
                            yield cleaned
               
                    
        except Exception as e:
            logger.error(f"[Gemini] Streaming error: {e}")
            raise



class LLMManagerFactory:
    
    @staticmethod
    async def create_llm() -> InferenceProvider:
        llm = GeminiLLMManager()
        await llm.initialize()
        return llm

