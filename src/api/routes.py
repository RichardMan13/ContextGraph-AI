"""
src/api/routes.py
─────────────────
Definition of FastAPI REST endpoints for the GraphRAG service.
Provides programmatic access to the underlying graphrag_chain.
"""

import sys
import subprocess
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.chains.graphrag_chain import get_graphrag_chain

router = APIRouter()

# Instantiate the singleton chain for performance
graphrag_chain = get_graphrag_chain()


class RecommendRequest(BaseModel):
    query: str


class RecommendResponse(BaseModel):
    answer: str
    cypher_query: Optional[str] = None
    candidate_ids: Optional[List[str]] = None
    source_documents: Optional[List[Dict[str, Any]]] = None


@router.post(
    "/v1/recommend",
    response_model=RecommendResponse,
    summary="Query your Personal Movie Archive",
)
async def recommend_endpoint(request: RecommendRequest):
    """
    Executes a semantic movie search using the Apache AGE + pgvector GraphRAG pipeline.
    Expects a natural language query string.
    """
    try:
        result = graphrag_chain.invoke({"query": request.query})

        docs = []
        for doc in result.get("source_documents", []):
            docs.append({"page_content": doc.page_content, "metadata": doc.metadata})

        return RecommendResponse(
            answer=result["answer"],
            cypher_query=result.get("cypher_query"),
            candidate_ids=result.get("candidate_ids"),
            source_documents=docs,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/ingest", summary="Trigger the ETL ingestion processes")
async def ingest_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers the Apache AGE and pgvector ingestion scripts asynchronously.
    """

    def run_ingestion():
        import logging

        logging.info("Starting API-triggered ingestion...")
        subprocess.run([sys.executable, "src/data/ingest_graph.py"], check=False)
        subprocess.run([sys.executable, "src/data/generate_embeddings.py"], check=False)
        logging.info("API-triggered ingestion complete.")

    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion process started in the background."}


@router.get("/v1/health", summary="API Health Check")
async def health_endpoint():
    """
    Returns the health status of the API.
    """
    return {"status": "ok", "service": "CineGraph-AI API"}
