"""
src/tools/graph_retriever.py
────────────────────────────
Executes generated Cypher queries on the Apache AGE graph and returns candidate IDs.
"""

import os
import logging
from typing import List, Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GraphRetriever:
    """A tool to execute Cypher strings against the AGE graph securely."""

    def __init__(self):
        self.db_params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5434)),
            "dbname": os.getenv("POSTGRES_DB", "cinegraph_db"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        }
        
    def _get_connection(self):
        conn = psycopg2.connect(**self.db_params)
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        conn.commit()
        return conn

    def retrieve_candidate_ids(self, cypher_query: str) -> Optional[List[str]]:
        """
        Executes the cypher query and returns matching movie IDs (const).
        If the query contains the semantic sentinel, returns `None` to bypass filter.
        """
        cleaned_query = cypher_query.strip()
        
        if not cleaned_query or "__NO_GRAPH_FILTER__" in cleaned_query:
            logger.info("GraphRetriever: Semantic query only (__NO_GRAPH_FILTER__). Bypassing edge filter.")
            return None
            
        conn = self._get_connection()
        candidate_ids = []
        try:
            with conn.cursor() as cur:
                logger.info("Executing Cypher inside Graph stage...")
                cur.execute(cleaned_query)
                rows = cur.fetchall()
                
                # Cypher results map back to psycopg2 strings wrapped in double quotes 
                # because they are `agtype` objects under the hood. e.g.: '"tt0268978"'
                for row in rows:
                    if len(row) > 0 and row[0]:
                        const_id = str(row[0]).strip('"')
                        candidate_ids.append(const_id)
                        
            logger.info("GraphRetriever: Found %d candidate IDs in the structural graph.", len(candidate_ids))
            return list(set(candidate_ids))  # deduplicate IDs safely
            
        except psycopg2.Error as e:
            logger.error("Graph query syntax evaluation failed. Error: %s", e)
            return []  # Return empty list properly denoting "0 matches" rather than complete fallback
        finally:
            conn.close()
