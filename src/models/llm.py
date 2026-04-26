"""
src/models/llm.py
─────────────────
Singleton provider for LangChain LLM instances configured for the GraphRAG pipeline.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

load_dotenv()

# Enable fast, process-local caching to save tokens on repeated completions
set_llm_cache(InMemoryCache())

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns the default ChatOpenAI instance for the application.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )
