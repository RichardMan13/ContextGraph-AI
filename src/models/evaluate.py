import json
import os
import sys
import asyncio
import mlflow
from dotenv import load_dotenv
from datasets import Dataset

# Fix import path so 'src' module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ragas import evaluate  # noqa: E402

try:
    from ragas.metrics.collections import (  # noqa: E402
        faithfulness,
        answer_relevancy,
        context_precision,
    )
except ImportError:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision  # noqa: E402

# Ensure env vars are loaded (OpenAI keys, MLflow, etc.)
load_dotenv()

from src.chains.graphrag_chain import get_graphrag_chain  # noqa: E402


async def run_eval():
    dataset_path = "references/golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Erro: {dataset_path} não encontrado.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    chain = get_graphrag_chain()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Gerando respostas para o dataset dourado ({len(golden_data)} itens)...")
    for item in golden_data:
        q = item["question"]
        print(f"Processando: {q}")

        try:
            result = await chain.ainvoke({"query": q})
            answer = result.get("answer", "")
            docs = [doc.page_content for doc in result.get("source_documents", [])]
        except Exception as e:
            print(f"Erro ao processar query '{q}': {e}")
            answer = ""
            docs = []

        questions.append(q)
        answers.append(answer)
        contexts.append(docs)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print(
        "\nIniciando avaliação RAGas (isso pode demorar e consumir requisições da OpenAI)..."
    )
    try:
        try:
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

            metrics_list = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
        except ImportError:
            metrics_list = (
                [faithfulness(), answer_relevancy(), context_precision()]
                if callable(faithfulness)
                else [faithfulness, answer_relevancy, context_precision]
            )

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        eval_llm = ChatOpenAI(model="gpt-4o-mini")
        eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            ragas_llm = LangchainLLMWrapper(eval_llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
        except ImportError:
            ragas_llm = eval_llm
            ragas_embeddings = eval_embeddings

        eval_result = evaluate(
            dataset, metrics=metrics_list, llm=ragas_llm, embeddings=ragas_embeddings
        )
        print("\n=== Resultados RAGas ===")
        print(eval_result)

        # Log to MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("CineGraph-AI_RAGas_Evaluation")

        with mlflow.start_run(run_name="RAGas_OneShot_Eval"):
            metrics_dict = {}
            import ast

            try:
                # Ragas EvaluationResult might not support direct dict access in newer versions,
                # but it prints perfectly as a dictionary string.
                res_str = str(eval_result).replace("nan", "0.0")
                eval_dict = ast.literal_eval(res_str)
                for k in ["faithfulness", "answer_relevancy", "context_precision"]:
                    metrics_dict[k] = float(eval_dict.get(k, 0.0))
            except Exception as ex:
                print(f"Aviso: Erro ao extrair métricas do eval_result: {ex}")
                metrics_dict = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                }

            mlflow.log_metrics(metrics_dict)

            # Log dataset questions and generated answers as a json artifact
            eval_output = {
                "questions": questions,
                "answers": answers,
                "ground_truths": ground_truths,
                "metrics": metrics_dict,
            }
            with open("eval_results_temp.json", "w", encoding="utf-8") as f:
                json.dump(eval_output, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact("eval_results_temp.json")
            os.remove("eval_results_temp.json")

        print("\nMétricas e logs salvos no MLflow com sucesso!")

    except Exception as e:
        print(f"Falha na avaliação do RAGas: {e}")


if __name__ == "__main__":
    asyncio.run(run_eval())
