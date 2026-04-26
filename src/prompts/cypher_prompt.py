"""
src/prompts/cypher_prompt.py
────────────────────────────
System prompt for the Graph Query Chain (Stage 1).

Injects a static schema block + 3 few-shot examples so gpt-4o-mini generates
valid Apache AGE Cypher without hallucinating node labels or property names.

Apache AGE Cypher specifics:
  - Queries are wrapped in: SELECT * FROM cypher('movies_graph', $$ ... $$) AS (col agtype)
  - String comparisons are case-sensitive by default
  - Property access: node.property (dot notation)
  - The RETURN clause must alias each column that appears in the AS (...) clause
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Static schema block ────────────────────────────────────────────────────────
# Injected verbatim into every system message so the LLM never guesses labels.

_SCHEMA = """
=== GRAPH SCHEMA (Apache AGE, graph name: movies_graph) ===

NODE LABELS & PROPERTIES:
  Movie    : const (TEXT, IMDb ID e.g. tt0268978), title (TEXT), imdb_rating (FLOAT),
             runtime_mins (FLOAT), num_votes (INT), release_date (TEXT)
  Director : name (TEXT)
  Genre    : name (TEXT)
  Year     : value (INT)

RELATIONSHIPS:
  (Movie)-[:DIRECTED_BY]->(Director)
  (Movie)-[:IN_GENRE]->(Genre)
  (Movie)-[:RELEASED_IN]->(Year)

APACHE AGE CYPHER SYNTAX RULES:
  1. Always wrap queries in:
       SELECT * FROM cypher('movies_graph', $$
           <CYPHER HERE>
       $$) AS (<alias> agtype);
  2. Use exact label names (Movie, Director, Genre, Year) — case-sensitive.
  3. Use exact property names as listed above — case-sensitive.
  4. String matching is case-sensitive. Genre names are in Portuguese as stored in the dataset:
       ['Animação', 'Aventura', 'Ação', 'Biografia', 'Comédia', 'Documentário',
        'Drama', 'Esportes', 'Família', 'Fantasia', 'Faroeste', 'Ficção científica',
        'Guerra', 'História', 'Mistério', 'Musical', 'Policial', 'Romance',
        'Suspense', 'Terror']
  5. To return multiple columns, list them all in the AS clause:
       $$) AS (const agtype, title agtype, imdb_rating agtype);
  6. Use LIMIT to cap results (default: 20).
  7. Do NOT use OPTIONAL MATCH unless explicitly needed.
  8. Director names use natural-language casing (e.g. 'Christopher Nolan').
"""

# ── Few-shot examples ──────────────────────────────────────────────────────────
# 3 patterns covering the main user query shapes.

_EXAMPLES = """
=== FEW-SHOT EXAMPLES ===

--- Example 1: Filter by director + genre + minimum rating ---
User: Sci-fi movies directed by Christopher Nolan rated above 8.0
Cypher:
SELECT * FROM cypher('movies_graph', $$
    MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director {{name: 'Christopher Nolan'}})
    MATCH (m)-[:IN_GENRE]->(g:Genre {{name: 'Ficção científica'}})
    WHERE m.imdb_rating >= 8.0
    RETURN m.const, m.title, m.imdb_rating
    ORDER BY m.imdb_rating DESC
    LIMIT 20
$$) AS (const agtype, title agtype, imdb_rating agtype);

--- Example 2: Filter by year range + genre ---
User: Horror movies from the 1980s
Cypher:
SELECT * FROM cypher('movies_graph', $$
    MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {{name: 'Terror'}})
    MATCH (m)-[:RELEASED_IN]->(y:Year)
    WHERE y.value >= 1980 AND y.value <= 1989
    RETURN m.const, m.title, m.imdb_rating
    ORDER BY m.imdb_rating DESC
    LIMIT 20
$$) AS (const agtype, title agtype, imdb_rating agtype);

--- Example 3: Well-rated movies (no genre/director filter — broad semantic query) ---
User: Best movies of all time
Cypher:
SELECT * FROM cypher('movies_graph', $$
    MATCH (m:Movie)
    WHERE m.imdb_rating >= 8.5
    RETURN m.const, m.title, m.imdb_rating
    ORDER BY m.imdb_rating DESC
    LIMIT 20
$$) AS (const agtype, title agtype, imdb_rating agtype);
"""

# ── Fallback instruction ───────────────────────────────────────────────────────
_FALLBACK_RULE = """
=== FALLBACK RULE ===
If the user query is purely semantic (no director, genre, year, or rating filter),
return the following sentinel value instead of a Cypher query:

  CYPHER: __NO_GRAPH_FILTER__

This tells the orchestrator to skip the graph step and run pure vector search.
"""

# ── Full system prompt ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = f"""You are an expert Apache AGE Cypher query generator for a movie recommendation system.

Your ONLY job is to translate the user's natural language query into a valid Apache AGE Cypher query
that returns a list of IMDb IDs (const) matching the user's filters.

{_SCHEMA}

{_EXAMPLES}

{_FALLBACK_RULE}

IMPORTANT RULES:
- Return ONLY the raw SQL/Cypher block. No explanation, no markdown fences.
- Never reference properties that are not listed in the schema.
- Always include m.const in the RETURN clause (it is the JOIN key for the vector search step).
- Keep LIMIT at 20 unless the user explicitly asks for more or fewer results.
"""

# ── LangChain ChatPromptTemplate ──────────────────────────────────────────────
cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)

__all__ = ["cypher_prompt"]
