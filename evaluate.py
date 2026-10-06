"""Score the RAG pipeline with DeepEval, using a local Ollama model as the judge.

Edit eval_questions.json with questions (and optional expected answers) about your PDFs.
Usage: python evaluate.py
"""
import json
import os

# A local judge is slow and handles one request at a time, so DeepEval's default
# per-task timeout cancels the run. Must be set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

import config
from rag import RAG


def main():
    with open(config.EVAL_FILE, encoding="utf-8") as f:
        items = json.load(f)

    rag = RAG()
    test_cases = []
    for item in items:
        result = rag.answer(item["question"])
        test_cases.append(LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item.get("expected_answer"),
            retrieval_context=result["contexts"],
        ))

    judge = OllamaModel(model=config.JUDGE_MODEL, base_url=config.OLLAMA_URL, temperature=0)
    metrics = [
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        FaithfulnessMetric(model=judge, threshold=0.7),
        ContextualRelevancyMetric(model=judge, threshold=0.5),
    ]
    evaluate(test_cases=test_cases, metrics=metrics, async_config=AsyncConfig(run_async=False))


if __name__ == "__main__":
    main()
