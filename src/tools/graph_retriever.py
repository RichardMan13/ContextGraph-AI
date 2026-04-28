"""
src/tools/graph_retriever.py
────────────────────────────
Executes generated Cypher queries on the Apache AGE graph and returns candidate IDs.
"""

import os
import logging
from typing import Dict, Any

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
            cur.execute('SET search_path = ag_catalog, "$user", public;')
        conn.commit()
        return conn

    def retrieve_candidate_ids(self, cypher_query: str) -> Dict[str, Any]:
        """
        Executes the cypher query.
        Returns a dict:
            - "ids": List of matching movie IDs (const)
            - "graph_context": A formatted string of ALL returned data for the LLM.
        """
        cleaned_query = cypher_query.strip()

        if not cleaned_query or "__NO_GRAPH_FILTER__" in cleaned_query:
            return {"ids": None, "graph_context": ""}

        conn = self._get_connection()
        candidate_ids = []
        rows_data = []
        try:
            with conn.cursor() as cur:
                logger.info("Executing Cypher inside Graph stage...")
                cur.execute(cleaned_query)

                # Get column names if possible (psycopg2 description)
                col_names = (
                    [desc[0] for desc in cur.description] if cur.description else []
                )

                rows = cur.fetchall()

                for row in rows:
                    row_parts = []
                    for idx, val in enumerate(row):
                        # Clean agtype wrappers
                        clean_val = str(val).strip('"')

                        # Heuristic: If it looks like an IMDb ID, add to candidates
                        if clean_val.startswith("tt") and len(clean_val) > 7:
                            candidate_ids.append(clean_val)

                        col_label = (
                            col_names[idx] if idx < len(col_names) else f"col_{idx}"
                        )
                        row_parts.append(f"{col_label}: {clean_val}")

                    rows_data.append(" | ".join(row_parts))

            context_str = "\n".join([f"- {r}" for r in rows_data])

            logger.info(
                "GraphRetriever: Found %d candidate IDs and %d data rows.",
                len(candidate_ids),
                len(rows_data),
            )

            return {
                "ids": list(set(candidate_ids)) if candidate_ids else [],
                "graph_context": context_str,
            }

        except psycopg2.Error as e:
            logger.error("Graph query syntax evaluation failed. Error: %s", e)
            return {"ids": [], "graph_context": "Error executing graph query."}
        finally:
            conn.close()
