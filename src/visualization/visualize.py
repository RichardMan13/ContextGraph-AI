"""
src/visualization/visualize.py
──────────────────────────────
Comprehensive visualization suite for CineGraph-AI.
Generates:
  1. Interactive Graph (pyvis)
  2. Consumption Stats (plotly)
  3. 2D Semantic Map (t-SNE + plotly)
  4. 3D Semantic Map (t-SNE 3D)
  5. Clustered Map (K-Means)
  6. Genre Heatmap (Density)
  7. Temporal Map (By Decade)
  8. Similarity Network (KNN Graph)
"""

import os
import logging
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

# Visualization libs
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pgvector.psycopg2 import register_vector

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_NAME = os.getenv("POSTGRES_GRAPH_PATH", "movies_graph")


def _get_conn():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5434)),
        dbname=os.getenv("POSTGRES_DB", "cinegraph_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )
    register_vector(conn)
    return conn


def _cypher(conn, query: str):
    with conn.cursor() as cur:
        cur.execute("LOAD 'age';")
        cur.execute('SET search_path = ag_catalog, "$user", public;')
        cur.execute(query)
        try:
            return cur.fetchall()
        except Exception:
            return []


def _fetch_data():
    """Shared data fetching for all semantic maps."""
    conn = _get_conn()
    query = """
    SELECT const, plot, embedding,
           metadata->>'title' as title,
           metadata->>'imdb_rating' as rating,
           metadata->>'year' as year,
           metadata->>'genres' as genres
    FROM public.movie_embeddings
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        conn.close()
        return None

    data = []
    for r in rows:
        const, plot, embedding, title, rating, year, genres = r
        data.append(
            {
                "title": title,
                "movie_id": const,
                "plot": plot,
                "rating": rating,
                "year": year,
                "genres": genres,
                "embedding": embedding,
            }
        )

    df = pd.DataFrame(data)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    conn.close()

    # Filter out movies with missing embeddings
    df = df[df["embedding"].notnull()].reset_index(drop=True)

    # Handle missing years/ratings without dropping the whole row
    df["year_display"] = df["year"].apply(lambda x: str(x) if x > 1800 else "Unknown")
    df["rating_display"] = df["rating"].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
    df["display_name"] = df["title"].fillna("Movie")

    initial_count = len(df)
    logger.info(
        f"Fetched {initial_count} movies. After filtering (embeddings only), {len(df)} remain."
    )

    if df.empty:
        logger.warning("No movies with embeddings found in public.movie_embeddings.")
        return None

    return df


# ── 1. Graph Connections ───────────────────────────────────────────────────────


def generate_graph_viz():
    logger.info("Generating Interactive Graph Visualization...")
    conn = _get_conn()
    query = f"""
    SELECT * FROM cypher('{GRAPH_NAME}', $$
        MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)
        MATCH (m)-[:IN_GENRE]->(g:Genre)
        RETURN m.title, d.name, g.name
        LIMIT 300
    $$) AS (m_title agtype, d_name agtype, g_name agtype);
    """
    rows = _cypher(conn, query)
    conn.close()

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="white",
        notebook=False,
    )
    net.force_atlas_2based()

    for m, d, g in rows:
        m_t, d_n, g_n = m.strip('"'), d.strip('"'), g.strip('"')
        # Clean label: hide if it looks like an ID
        m_label = "" if m_t.startswith("tt") else m_t
        net.add_node(
            m_t,
            label=m_label,
            title=f"Movie: {m_t}",
            color="#bb86fc",
            size=25,
            border_width=2,
        )
        net.add_node(
            d_n,
            label=d_n,
            title=f"Director: {d_n}",
            color="#03dac6",
            size=20,
            shape="diamond",
        )
        net.add_node(
            g_n, label=g_n, title=f"Genre: {g_n}", color="#ff7597", size=18, shape="dot"
        )
        net.add_edge(m_t, d_n, color="#555555", width=1.5)
        net.add_edge(m_t, g_n, color="#555555", width=1.5)

    net.save_graph(str(OUTPUT_DIR / "movie_graph.html"))


# ── 2. Consumption Stats ───────────────────────────────────────────────────────


def generate_stats_viz():
    logger.info("Generating Consumption Stats...")
    conn = _get_conn()

    # Genre Distribution
    query_genres = f"SELECT * FROM cypher('{GRAPH_NAME}', $$ MATCH (g:Genre)<-[:IN_GENRE]-(m:Movie) RETURN g.name, count(m) ORDER BY count(m) DESC $$) AS (name agtype, count agtype);"
    df_genres = pd.DataFrame(_cypher(conn, query_genres), columns=["Genre", "Count"])
    df_genres["Genre"] = df_genres["Genre"].str.strip('"')

    fig = px.treemap(
        df_genres,
        path=["Genre"],
        values="Count",
        title="Genre Treemap",
        color_continuous_scale="Magma",
    )
    fig.update_layout(template="plotly_dark")
    fig.write_html(str(OUTPUT_DIR / "genre_stats.html"))

    # Rating Distribution
    query_ratings = f"SELECT * FROM cypher('{GRAPH_NAME}', $$ MATCH (m:Movie) WHERE m.imdb_rating IS NOT NULL RETURN m.imdb_rating $$) AS (rating agtype);"
    df_ratings = pd.DataFrame(_cypher(conn, query_ratings), columns=["Rating"])
    df_ratings["Rating"] = df_ratings["Rating"].astype(float)

    fig = px.histogram(
        df_ratings,
        x="Rating",
        nbins=20,
        title="Rating Distribution",
        color_discrete_sequence=["#03dac6"],
    )
    fig.update_layout(template="plotly_dark")
    fig.write_html(str(OUTPUT_DIR / "rating_stats.html"))

    conn.close()


# ── 3. Semantic Maps ───────────────────────────────────────────────────────────


def generate_semantic_suite(df):
    if df is None or df.empty:
        logger.warning("Skipping Semantic Suite: No valid data available.")
        return

    try:
        embeddings_list = df["embedding"].tolist()
        if not embeddings_list:
            logger.warning("Skipping Semantic Suite: Embedding list is empty.")
            return
        X = np.stack(embeddings_list)
    except Exception as e:
        logger.error(f"Failed to stack embeddings: {e}")
        return

    # 3.1 2D Map (t-SNE)
    logger.info("Generating 2D Semantic Map...")
    tsne_2d = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=42)
    X_2d = tsne_2d.fit_transform(X)
    df["x2d"], df["y2d"] = X_2d[:, 0], X_2d[:, 1]

    fig = px.scatter(
        df,
        x="x2d",
        y="y2d",
        hover_name="display_name",
        color="rating",
        size="rating",
        size_max=12,
        hover_data={
            "year_display": True,
            "rating_display": True,
            "genres": True,
            "rating": False,
            "x2d": False,
            "y2d": False,
            "display_name": False,
        },
        title="2D Semantic Movie Map (Plot Similarity)",
        color_continuous_scale="Viridis",
        height=800,
    )

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="white")))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Semantic Dimension 1",
        yaxis_title="Semantic Dimension 2",
        coloraxis_colorbar=dict(title="IMDb Rating"),
    )
    fig.write_html(str(OUTPUT_DIR / "semantic_map_2d.html"))

    # 3.2 3D Map
    logger.info("Generating 3D Semantic Map...")
    tsne_3d = TSNE(n_components=3, perplexity=min(30, len(X) - 1), random_state=42)
    X_3d = tsne_3d.fit_transform(X)
    df["x3d"], df["y3d"], df["z3d"] = X_3d[:, 0], X_3d[:, 1], X_3d[:, 2]

    fig = px.scatter_3d(
        df,
        x="x3d",
        y="y3d",
        z="z3d",
        color="rating",
        size="rating",
        size_max=10,
        hover_name="display_name",
        hover_data={
            "year_display": True,
            "rating_display": True,
            "genres": True,
            "rating": False,
            "x3d": False,
            "y3d": False,
            "z3d": False,
            "display_name": False,
        },
        title="3D Semantic Movie Map (Plot Clusters)",
        color_continuous_scale="Plasma",
        opacity=0.7,
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
    )
    fig.write_html(str(OUTPUT_DIR / "semantic_map_3d.html"))

    # 3.3 Clustered Map (K-Means)
    logger.info("Generating Clustered Map...")
    # Heuristic: 5 clusters for visibility
    k = min(5, len(df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X).astype(str)

    fig = px.scatter(
        df,
        x="x2d",
        y="y2d",
        color="cluster",
        hover_name="display_name",
        hover_data={
            "year_display": True,
            "rating_display": True,
            "genres": True,
            "cluster": False,
            "x2d": False,
            "y2d": False,
            "display_name": False,
        },
        title=f"Movie Clusters (k={k}) - Semantic Grouping by Plot Similarity",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_traces(
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color="white"))
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Semantic Dimension 1",
        yaxis_title="Semantic Dimension 2",
    )
    fig.write_html(str(OUTPUT_DIR / "semantic_clusters.html"))

    # Generate Cluster Report
    report_path = OUTPUT_DIR / "cluster_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"CINEGRAPH-AI CLUSTER REPORT (k={k})\n")
        f.write("=" * 40 + "\n\n")
        for i in range(k):
            cluster_movies = df[df["cluster"] == str(i)]
            f.write(f"Cluster {i} ({len(cluster_movies)} movies):\n")
            # Show top 5 movies by rating in this cluster
            top_movies = cluster_movies.sort_values("rating", ascending=False).head(8)
            for _, m in top_movies.iterrows():
                f.write(f"  - [{m['rating']}] {m['title']} ({int(m['year'])})\n")
            f.write("\n")
    logger.info("✅ Cluster report saved to: %s", report_path)

    # 3.4 Genre Heatmap
    logger.info("Generating Genre Density Map...")
    # We pick the top 5 genres to avoid clutter
    top_genres = ["Drama", "Ação", "Comédia", "Terror", "Ficção científica"]
    fig = go.Figure()
    for g in top_genres:
        mask = df["genres"].str.contains(g, na=False)
        if mask.any():
            fig.add_trace(
                go.Histogram2dContour(
                    x=df[mask]["x2d"],
                    y=df[mask]["y2d"],
                    name=g,
                    colorscale="Blues" if g == "Drama" else "Reds",
                    showscale=False,
                    ncontours=10,
                    opacity=0.5,
                )
            )
    fig.add_trace(
        go.Scatter(
            x=df["x2d"],
            y=df["y2d"],
            mode="markers",
            marker=dict(color="white", size=2, opacity=0.3),
        )
    )
    fig.update_layout(title="Genre Density in Semantic Space", template="plotly_dark")
    fig.write_html(str(OUTPUT_DIR / "genre_heatmap.html"))

    # 3.5 Temporal Map
    logger.info("Generating Temporal Evolution Map...")
    df["decade"] = df["year"].apply(
        lambda x: f"{(x // 10 * 10)}s" if x > 1800 else "Unknown"
    )
    # Sort by decade to ensure legend order
    df = df.sort_values("year")

    fig = px.scatter(
        df,
        x="x2d",
        y="y2d",
        color="decade",
        hover_name="display_name",
        hover_data={
            "year_display": True,
            "rating_display": True,
            "decade": False,
            "x2d": False,
            "y2d": False,
            "display_name": False,
        },
        title="Movie Distribution by Decade (Semantic Shift)",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        marginal_x="box",
        marginal_y="violin",
    )

    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Semantic Dimension 1",
        yaxis_title="Semantic Dimension 2",
    )
    # 3.6 Rating Trend (New Comparative Intent)
    logger.info("Generating Rating vs. Year Trend...")
    # Filter out year 0 for the trend line
    df_trend = df[df["year"] > 1800].groupby("year")["rating"].mean().reset_index()
    if not df_trend.empty:
        fig = px.line(
            df_trend,
            x="year",
            y="rating",
            title="Average Movie Rating Over Years",
            labels={"rating": "Avg IMDb Rating", "year": "Year of Release"},
            template="plotly_dark",
            color_discrete_sequence=["#03dac6"],
        )
        fig.add_bar(
            x=df_trend["year"], y=df_trend["rating"], opacity=0.3, name="Rating Bar"
        )
        fig.write_html(str(OUTPUT_DIR / "rating_trend.html"))
    else:
        logger.warning("Skipping Rating Trend: No valid year data available.")


# ── 4. Similarity Network ──────────────────────────────────────────────────────


def generate_similarity_network(df):
    logger.info("Generating Similarity Network (KNN Graph)...")
    X = np.stack(df["embedding"].values)
    sim_matrix = cosine_similarity(X)

    # We only take the top 100 movies to keep the graph readable
    top_n = min(100, len(df))
    subset_df = df.iloc[:top_n].copy()
    subset_sim = sim_matrix[:top_n, :top_n]

    net = Network(height="800px", width="100%", bgcolor="#1a1a1a", font_color="white")
    net.toggle_physics(True)

    for i, row in subset_df.iterrows():
        title_val = str(row["display_name"])
        # Hide label if it's just a placeholder or ID
        node_label = (
            "" if (title_val == "Movie" or title_val.startswith("tt")) else title_val
        )
        net.add_node(
            i,
            label=node_label,
            title=f"{title_val}\n\n{row['plot'][:200]}...",
            color="#03dac6",
        )

    # Connect each movie to its top 3 most similar neighbors (if similarity > 0.8)
    for i in range(top_n):
        # Get indices of top 4 (including self)
        indices = np.argsort(subset_sim[i])[-4:-1]
        for idx in indices:
            if subset_sim[i][idx] > 0.8:
                net.add_edge(int(i), int(idx), value=float(subset_sim[i][idx]))

    net.save_graph(str(OUTPUT_DIR / "similarity_network.html"))


if __name__ == "__main__":
    try:
        df = _fetch_data()
        if df is not None:
            generate_graph_viz()
            generate_stats_viz()
            generate_semantic_suite(df)
            generate_similarity_network(df)
            logger.info("\n🚀 ALL 8 visualizations generated in reports/figures/")
        else:
            logger.error("No data found to visualize.")
    except Exception:
        logger.exception("Visualization failed:")
