import logging
from pathlib import Path

from dotenv import load_dotenv

# Importa as chains do projeto
from src.chains.graph_chain import get_graph_chain
from src.chains.graphrag_chain import get_graphrag_chain
from src.chains.vector_chain import get_vector_chain

# Configura o log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_chains():
    """
    Gera e exporta a visualização da arquitetura das chains do LangChain (LCEL).
    Salva os grafos Mermaid em reports/figures e imprime em ASCII no terminal.
    """
    # Carrega variáveis de ambiente
    load_dotenv()

    # Define o diretório de saída
    ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = ROOT / "reports" / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Instanciando as chains para visualização...")

    # Obtém as chains (Nota: elas precisam apenas ser construídas, não executadas)
    graphrag_chain = get_graphrag_chain()
    graph_chain = get_graph_chain()
    vector_chain = get_vector_chain()

    chains_to_visualize = {
        "graphrag_full_chain": graphrag_chain,
        "stage1_graph_chain": graph_chain,
        "stage2_vector_chain": vector_chain,
    }

    for name, chain in chains_to_visualize.items():
        logger.info(f"\n--- Visualizando: {name} ---")
        try:
            # Extrai o grafo estrutural da chain
            grafo = chain.get_graph()

            # 1. Imprime versão ASCII no terminal
            print(f"\n[ASCII Graph - {name}]\n")
            try:
                grafo.print_ascii()
            except Exception as e:
                logger.warning(
                    f"Não foi possível imprimir o grafo ASCII (instale grandalf): {e}"
                )

            # 2. Gera a sintaxe Mermaid
            mermaid_syntax = grafo.draw_mermaid()

            # Salva o arquivo Mermaid (.md)
            output_file = OUTPUT_DIR / f"{name}_architecture.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Arquitetura LCEL: {name}\n\n")
                f.write("```mermaid\n")
                f.write(mermaid_syntax)
                f.write("\n```\n")

            logger.info(f"✅ Grafo salvo em formato Mermaid: {output_file}")

        except Exception as e:
            logger.error(f"Erro ao visualizar a chain {name}: {e}")


if __name__ == "__main__":
    visualize_chains()
