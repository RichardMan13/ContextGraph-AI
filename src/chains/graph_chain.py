"""
src/chains/graph_chain.py
─────────────────────────
Stage 1: Generates a Cypher query from natural language using the Schema 
Prompt and outputs a structured Pydantic object.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.models.llm import get_llm
from src.prompts.cypher_prompt import cypher_prompt
from src.tools.graph_retriever import GraphRetriever

class GraphQueryResult(BaseModel):
    cypher_query: str = Field(
        description="The strictly valid Apache AGE Cypher query matching the user goal, or the exact string '__NO_GRAPH_FILTER__' if no graph metadata boundaries apply."
    )

def _execute_cypher(result: GraphQueryResult) -> Dict[str, Any]:
    """
    Executes the structured query and returns the candidate IDs.
    Returns a dict with state to pass along to Stage 2.
    """
    retriever = GraphRetriever()
    candidate_ids = retriever.retrieve_candidate_ids(result.cypher_query)
    
    return {
        "cypher_query": result.cypher_query,
        "candidate_ids": candidate_ids
    }

def get_graph_chain():
    """
    Returns an LCEL chain that:
      1. Formats the user question into the CYPHER prompt.
      2. Generates structured Cypher output via GPT-4o-mini.
      3. Executes the string on the DB intercepting the graph nodes. 
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(GraphQueryResult)
    
    chain = cypher_prompt | structured_llm | RunnableLambda(_execute_cypher)
    return chain

if __name__ == "__main__":
    # Test script block
    chain = get_graph_chain()
    print(chain.invoke({"query": "Quero um filme de Ficção científica de Christopher Nolan, com nota acima de 8.0"}))
