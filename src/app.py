"""
src/app.py
──────────
FastAPI application entrypoint.
Mounts REST logic (from api/routes.py) and the Gradio Web UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from src.api.routes import router as api_router
from src.ui.blocks import create_ui

app = FastAPI(
    title="CineGraph-AI API",
    description="Knowledge Graph RAG for Movie Recommendations using Apache AGE and pgvector",
    version="1.0.0"
)

# Enable CORS for frontend integrations if necessary
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount pure backend REST API logic under /api
app.include_router(api_router, prefix="/api", tags=["Core System"])

# 2. Mount Gradio interface directly onto the root path (/)
demo = create_ui()
# Enable Queuing as explicitly requested before launching/mounting
demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=7860, reload=True)
