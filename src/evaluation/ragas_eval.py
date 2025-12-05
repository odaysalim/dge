"""
RAGAs Evaluation Module for Agentic RAG System.

This module provides evaluation capabilities using the RAGAs framework:
- Faithfulness: Measures factual consistency of answers with context
- Answer Relevancy: Measures how relevant the answer is to the question
- Context Precision: Measures the signal-to-noise ratio in retrieved context
- Context Recall: Measures if all relevant info is retrieved

Supports both Ollama and OpenAI as evaluation LLMs.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.settings import (
    CONFIG, LLM_PROVIDER, EVAL_DATA_DIR,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL
)


@dataclass
class EvaluationSample:
    """Single evaluation sample with question, answer, context, and ground truth."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results from RAGAs evaluation."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    num_samples: int
    timestamp: str
    details: List[Dict[str, Any]]


class RAGEvaluator:
    """
    RAGAs-based evaluator for the Agentic RAG system.

    Uses RAGAs metrics to evaluate:
    - Faithfulness: Is the answer factually consistent with the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Is the retrieved context relevant?
    - Context Recall: Is all necessary information retrieved?
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the evaluator.

        Args:
            output_dir: Directory for saving evaluation results
        """
        self.output_dir = output_dir or EVAL_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import ragas
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from datasets import Dataset

            self.evaluate = evaluate
            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
            self.Dataset = Dataset
            self.ragas_available = True
            logger.info("RAGAs framework loaded successfully")

        except ImportError as e:
            logger.warning(f"RAGAs not available: {e}")
            logger.warning("Install with: pip install ragas datasets")
            self.ragas_available = False

        # Configure LLM for evaluation
        self._setup_eval_llm()

    def _setup_eval_llm(self):
        """Configure the LLM used for RAGAs evaluation."""
        if not self.ragas_available:
            return

        try:
            if LLM_PROVIDER == "ollama":
                from langchain_community.llms import Ollama
                from langchain_community.embeddings import OllamaEmbeddings

                self.llm = Ollama(
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0
                )
                self.embeddings = OllamaEmbeddings(
                    model=CONFIG['ollama']['embedding_model'],
                    base_url=OLLAMA_BASE_URL
                )
                logger.info(f"Using Ollama for evaluation: {OLLAMA_MODEL}")

            else:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings

                self.llm = ChatOpenAI(
                    model=OPENAI_MODEL,
                    api_key=OPENAI_API_KEY,
                    temperature=0
                )
                self.embeddings = OpenAIEmbeddings(
                    model=CONFIG['openai']['embedding_model'],
                    api_key=OPENAI_API_KEY
                )
                logger.info(f"Using OpenAI for evaluation: {OPENAI_MODEL}")

        except Exception as e:
            logger.error(f"Failed to setup evaluation LLM: {e}")
            self.ragas_available = False

    def evaluate_samples(
        self,
        samples: List[EvaluationSample]
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a list of samples using RAGAs metrics.

        Args:
            samples: List of EvaluationSample objects

        Returns:
            EvaluationResult with scores and details
        """
        if not self.ragas_available:
            logger.error("RAGAs is not available. Cannot evaluate.")
            return None

        if not samples:
            logger.error("No samples provided for evaluation")
            return None

        logger.info(f"Evaluating {len(samples)} samples...")

        # Prepare dataset
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples]
        }

        dataset = self.Dataset.from_dict(data)

        try:
            # Run RAGAs evaluation
            results = self.evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )

            # Extract scores
            df = results.to_pandas()
            scores = df.mean(numeric_only=True).to_dict()

            # Calculate overall score (average of all metrics)
            metric_scores = [
                scores.get("faithfulness", 0),
                scores.get("answer_relevancy", 0),
                scores.get("context_precision", 0),
                scores.get("context_recall", 0)
            ]
            overall = sum(metric_scores) / len(metric_scores) if metric_scores else 0

            # Prepare detailed results
            details = df.to_dict(orient="records")

            result = EvaluationResult(
                faithfulness=scores.get("faithfulness", 0),
                answer_relevancy=scores.get("answer_relevancy", 0),
                context_precision=scores.get("context_precision", 0),
                context_recall=scores.get("context_recall", 0),
                overall_score=overall,
                num_samples=len(samples),
                timestamp=datetime.utcnow().isoformat(),
                details=details
            )

            logger.info(f"Evaluation complete. Overall score: {overall:.2%}")

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_results(self, result: EvaluationResult, name: str = "eval"):
        """
        Save evaluation results to file.

        Args:
            result: EvaluationResult to save
            name: Base name for the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.output_dir / filename

        data = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "overall_score": result.overall_score,
            "num_samples": result.num_samples,
            "timestamp": result.timestamp,
            "details": result.details
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def load_test_set(self, filepath: Path) -> List[EvaluationSample]:
        """
        Load a test set from a JSON file.

        Expected format:
        [
            {
                "question": "What is...",
                "ground_truth": "The answer is...",
                "contexts": ["context1", "context2"]  // optional
            }
        ]
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = EvaluationSample(
                question=item["question"],
                answer=item.get("answer", ""),  # Will be filled by RAG system
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth")
            )
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} test samples from {filepath}")
        return samples

    def create_sample_test_set(self) -> Path:
        """
        Create a sample test set for evaluation.

        Returns:
            Path to the created test set file
        """
        samples = [
            {
                "question": "What is the annual leave entitlement for employees?",
                "ground_truth": "Annual leave entitlement varies based on grade and years of service, typically ranging from 22 to 30 working days per year."
            },
            {
                "question": "How do I create a purchase requisition in SAP Ariba?",
                "ground_truth": "To create a purchase requisition, log into SAP Ariba, navigate to the Requisition module, click Create, fill in the required fields including description and quantity, and submit for approval."
            },
            {
                "question": "What are the password requirements in the security policy?",
                "ground_truth": "Passwords must be at least 12 characters long, include uppercase and lowercase letters, numbers, and special characters, and must be changed every 90 days."
            },
            {
                "question": "What is the procurement threshold for direct purchases?",
                "ground_truth": "Direct purchases can be made for items under a certain threshold without requiring competitive bidding, as defined in the procurement policy."
            },
            {
                "question": "How many sick leave days are employees entitled to?",
                "ground_truth": "Employees are entitled to sick leave as specified in the HR bylaws, with full pay for a certain period followed by reduced pay."
            }
        ]

        filepath = self.output_dir / "sample_test_set.json"
        with open(filepath, "w") as f:
            json.dump(samples, f, indent=2)

        logger.info(f"Created sample test set at {filepath}")
        return filepath


def run_evaluation(
    test_file: Optional[str] = None,
    output_name: str = "evaluation"
) -> Optional[EvaluationResult]:
    """
    Run RAGAs evaluation on a test set.

    Args:
        test_file: Path to test set JSON file (creates sample if None)
        output_name: Name for the output results file

    Returns:
        EvaluationResult or None if evaluation fails
    """
    evaluator = RAGEvaluator()

    if not evaluator.ragas_available:
        logger.error("RAGAs is not available. Cannot run evaluation.")
        return None

    # Load or create test set
    if test_file:
        samples = evaluator.load_test_set(Path(test_file))
    else:
        # Create sample test set
        test_path = evaluator.create_sample_test_set()
        samples = evaluator.load_test_set(test_path)

    # Note: In real usage, you would run the RAG system to get answers
    # For now, we just demonstrate the evaluation structure
    logger.info("Note: Answers and contexts should be populated by running the RAG system")

    # Run evaluation
    result = evaluator.evaluate_samples(samples)

    if result:
        evaluator.save_results(result, output_name)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAs evaluation on the RAG system")
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test set JSON file"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="evaluation",
        help="Name for output results file"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample test set"
    )

    args = parser.parse_args()

    if args.create_sample:
        evaluator = RAGEvaluator()
        filepath = evaluator.create_sample_test_set()
        print(f"Created sample test set: {filepath}")
    else:
        result = run_evaluation(
            test_file=args.test_file,
            output_name=args.output_name
        )

        if result:
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Faithfulness:       {result.faithfulness:.2%}")
            print(f"Answer Relevancy:   {result.answer_relevancy:.2%}")
            print(f"Context Precision:  {result.context_precision:.2%}")
            print(f"Context Recall:     {result.context_recall:.2%}")
            print("-" * 60)
            print(f"Overall Score:      {result.overall_score:.2%}")
            print(f"Samples Evaluated:  {result.num_samples}")
            print("=" * 60)
