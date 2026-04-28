import json
import os
import sys
import asyncio
import mlflow
from dotenv import load_dotenv
from ragas import EvaluationDataset

# Fix import path so 'src' module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from openai import AsyncOpenAI
from ragas import aevaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Ensure env vars are loaded (OpenAI keys, MLflow, etc.)
load_dotenv()

from src.chains.graphrag_chain import get_graphrag_chain  # noqa: E402

if sys.platform == "win32":
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def run_eval():
    dataset_path = "references/golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    chain = get_graphrag_chain()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(
        f"Generating answers for the golden dataset ({len(golden_data)} items) in parallel..."
    )
    tasks = [chain.ainvoke({"query": item["question"]}) for item in golden_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item, result in zip(golden_data, results):
        q = item["question"]

        if isinstance(result, Exception):
            print(f"Error processing query '{q}': {result}")
            answer = ""
            docs = []
            result = {}
        else:
            answer = result.get("answer", "")
            docs = [doc.page_content for doc in result.get("source_documents", [])]

        questions.append(q)
        answers.append(answer)

        # Combine graph context and vector documents
        all_contexts = []
        graph_ctx = result.get("graph_context", "")
        if graph_ctx:
            all_contexts.append(f"Structured Graph Knowledge:\n{graph_ctx}")
        all_contexts.extend(docs)
        contexts.append(all_contexts)

        ground_truths.append(item["ground_truth"])

    # Use EvaluationDataset for compatibility
    dataset = EvaluationDataset.from_list(
        [
            {"user_input": q, "response": a, "retrieved_contexts": c, "reference": g}
            for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        ]
    )

    # Setup Evaluator Components
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=client)
    setattr(llm, "is_async", True)  # Modern Rule: Enable async paths

    embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", interface="legacy"
    )

    # Use initialized metrics from ragas.metrics
    metrics_list = [faithfulness, answer_relevancy, context_precision]

    print("\nStarting RAGas evaluation...")
    try:
        # Use aevaluate for production-safe async execution
        eval_result = await aevaluate(
            dataset=dataset, metrics=metrics_list, llm=llm, embeddings=embeddings
        )
        print("\n=== RAGas Results ===")
        print(eval_result)

        # Log to MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("CineGraph-AI_RAGas_Evaluation")

        with mlflow.start_run(run_name="RAGas_Modernized_Eval"):
            # Extract mean metrics from the evaluation result
            metrics_dict = eval_result.to_pandas().mean().to_dict()
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

        print("\nMetrics and logs successfully saved to MLflow!")

    except Exception as e:
        print(f"RAGas evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_eval())
