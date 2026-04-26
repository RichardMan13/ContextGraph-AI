"""
src/ui/blocks.py
────────────────
Gradio user interface obeying the specific split layout requirements:
- Left column: Chatbot streaming
- Right column: HTML cards dynamically generated from pgvector source documents.
"""

import gradio as gr
from typing import List, Dict, Any

from src.chains.graphrag_chain import get_graphrag_chain

graphrag_chain = get_graphrag_chain()

def generate_movie_cards(docs: List[Dict[str, Any]]) -> str:
    """Generates pure HTML cards from retrieved metadata."""
    if not docs:
        return "<div style='padding:20px; color:gray'>Nenhum filme encontrado no contexto atual.</div>"
    
    html = "<div style='display:flex; flex-direction:column; gap:15px; max-height: 600px; overflow-y: auto; padding-right: 10px;'>"
    for doc in docs:
        meta = doc.get("metadata", {})
        
        # Fallback values if enrichment missed something
        title = meta.get("title", meta.get("const", "ID Desconhecido"))
        rating = meta.get("imdb_rating", "N/A")
        year = meta.get("release_date", "N/A")
        length = meta.get("runtime_mins", "N/A")
        dist = meta.get("semantic_distance", 0.0)
        
        # Truncate plot
        plot_raw = doc.get("page_content", "")
        plot = plot_raw[:180] + "..." if len(plot_raw) > 180 else plot_raw
        
        card = f"""
        <div style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:15px; box-shadow:0 4px 6px rgba(0,0,0,0.3); color:#e0e0e0'>
            <h3 style='margin-top:0; color:#ba86fc'>🎥 {title} <span style='font-size:0.8em; color:#a0a0a0'>⭐ {rating}</span></h3>
            <p style='margin:5px 0; font-size:0.9em; color:#bbb'>📅 {year} | ⏳ {length}m | 🎯 Distância: {dist:.3f}</p>
            <p style='font-size:0.9em; font-style:italic'>{plot}</p>
        </div>
        """
        html += card
        
    html += "</div>"
    return html


async def process_chat(user_msg: str, history: list):
    """
    Handles streaming via async def and yield as requested.
    Invokes the GraphRAG chain, yields a simulated streaming response,
    then updates the right-column HTML cards.
    """
    history = history or []
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": "⏳ Explorando o Grafo de Conhecimento..."})
    yield history, gr.update()
    
    try:
        # Await the pipeline asynchronously
        result = await graphrag_chain.ainvoke({"query": user_msg})
        
        final_answer = result.get("answer", "Não foi possível gerar resposta.")
        docs = []
        for doc in result.get("source_documents", []):
            docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
        html_cards = generate_movie_cards(docs)
        
        # Provide simple word-by-word streaming simulation to satisfy yield UI UX
        words = final_answer.split(" ")
        history[-1]["content"] = ""
        for word in words:
            import asyncio
            history[-1]["content"] += word + " "
            yield history, html_cards
            await asyncio.sleep(0.01)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        history[-1]["content"] = f"**Erro Interno:** {str(e)}"
        yield history, gr.update()


def create_ui():
    """Builds the strict split-layout UI explicitly requested in the Phase 6 spec."""
    PREMIUM_CSS = """
    body, .gradio-container { background: #0f111a !important; color: #e2e8f0; font-family: 'Inter', sans-serif !important; }
    """
    
    theme = gr.themes.Monochrome()
    
    with gr.Blocks(theme=theme, css=PREMIUM_CSS, title="CineGraph-AI") as demo:
        gr.Markdown("# 🎬 CineGraph-AI")
        gr.Markdown("Recomendação com **Apache AGE (Grafo)**, **pgvector (Vetor)** e **LangChain**.")
        
        with gr.Row():
            # LEFT COLUMN: Chatbot streaming interface
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=550, show_label=False)
                with gr.Row():
                    txt = gr.Textbox(placeholder="Descreva o filme perfeito...", show_label=False, scale=8)
                    btn = gr.Button("Enviar", variant="primary", scale=2)
            
            # RIGHT COLUMN: HTML Movie Cards
            with gr.Column(scale=4):
                gr.Markdown("### 🗂️ Filmes Base Recuperados")
                cards_html = gr.HTML("<div style='padding:20px; color:gray; border:1px dashed #333; border-radius:10px;'>Inicie uma busca para visualizar os filmes selecionados pelo Grafo.</div>")
                
        # Connect listeners
        txt.submit(process_chat, inputs=[txt, chatbot], outputs=[chatbot, cards_html])
        btn.click(process_chat, inputs=[txt, chatbot], outputs=[chatbot, cards_html])
        
        # Clear textbox on submit implicitly
        txt.submit(lambda: "", inputs=None, outputs=txt, queue=False)
        btn.click(lambda: "", inputs=None, outputs=txt, queue=False)
        
    return demo
