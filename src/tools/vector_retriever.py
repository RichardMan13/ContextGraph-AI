"""
src/tools/vector_retriever.py
─────────────────────────────
Retrieves movies semantically stored in PostgreSQL using pgvector.
Supports hybrid filtering by passing `candidate_ids` derived from the Graph execution.
"""

import os
from typing import List, Optional, Any

import psycopg2
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class VectorRetriever:
    """Custom Retriever for PostgreSQL + pgvector using our stored procedure."""

    def __init__(self):
        # We initialise the embeddings model using the environment variable
        self.embedder = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Setup connection args
        self.db_params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5434)),
            "dbname": os.getenv("POSTGRES_DB", "cinegraph_db"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        }

    def _get_connection(self):
        """Returns a new psycopg2 connection."""
        return psycopg2.connect(**self.db_params)

    def search(self, query: str, candidate_ids: Optional[List[str]] = None, top_k: int = 10) -> List[Document]:
        """
        Submits the query to the OpenAI Embedding API, then executes the stored 
        procedure `search_movie_embeddings` to find the nearest movies.
        
        If candidate_ids is provided, it restricts semantic search ONLY to those IDs.
        """
        # 1. Generate Query Vector
        query_vector = self.embedder.embed_query(query)

        # 2. Search Database
        docs = []
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # The stored procedure handles the HNSW index search automatically.
                # Signature: search_movie_embeddings(query_embedding VECTOR, candidate_ids TEXT[], top_k INT)
                cur.execute(
                    "SELECT const, plot, metadata, distance FROM search_movie_embeddings(%s::vector, %s, %s)",
                    (query_vector, candidate_ids, top_k)
                )
                
                rows = cur.fetchall()
                for row in rows:
                    const, plot, metadata, distance = row
                    
                    # Convert to LangChain Document format for next stages
                    # We inject const and distance into the metadata
                    meta = metadata or {}
                    meta["const"] = const
                    meta["semantic_distance"] = distance

                    docs.append(Document(page_content=plot, metadata=meta))
                    
        finally:
            conn.close()

        return docs
