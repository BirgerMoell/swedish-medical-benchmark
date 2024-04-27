import json
import numpy as np

from functools import partial
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from run_llm.benchmark_set_up import Benchmark, get_benchmark_by_name
from collections import defaultdict


# Configuration
# =============
PATH = "results.json"  # Path to the results file
PRINT_ALL = False  # Print all groups or just the top 5
INCLUDE_ALL_METRICS_IN_PROPERTY = False  # Include all metrics in the property ranking
RANK_BY = "f1"  # This is due to the unbalanced data. Can be: "accuracy", "precision", "recall", "f1"
MIN_SAMPLES = 50  # Minimum number of samples to consider a group
SAVE_RESULT_PATH = (
    None  # If saving to file set it to something like: "eval_results.txt"
)
OUTPUT_FILE = open(SAVE_RESULT_PATH, "w") if SAVE_RESULT_PATH else None
AVERAGE_METHOD = None  # Can be "micro", "macro", "weighted", "samples", None

printo = partial(print, file=OUTPUT_FILE)


# Functions
# =========
def calculate_metrics(ground_truths: list[str], predictions: list[str]):
    metrics = {
        "accuracy": accuracy_score(ground_truths, predictions),
        "precision": precision_score(
            ground_truths, predictions, average=AVERAGE_METHOD, zero_division=0
        ),
        "recall": recall_score(
            ground_truths, predictions, average=AVERAGE_METHOD, zero_division=0
        ),
        "f1": f1_score(ground_truths, predictions, average=AVERAGE_METHOD),
        "confusion_matrix": confusion_matrix(ground_truths, predictions),
    }
    return {
        k: list(v) if isinstance(v, np.ndarray) and k != "confusion_matrix" else v
        for k, v in metrics.items()
    }


def print_metrics(metrics: dict):
    printo(f"Accuracy: {metrics['accuracy']}")
    printo(f"Precision: {metrics['precision']}")
    printo(f"Recall: {metrics['recall']}")
    printo(f"F1: {metrics['f1']}")
    printo(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


def get_groups_by_property(ids: list[str], property: str, benchmark: Benchmark):
    groups = defaultdict(list[int])
    data = benchmark.data if not hasattr(benchmark, "og_data") else benchmark.og_data
    for n, id_ in enumerate(ids):
        if not isinstance(data[id_][property], list):
            data[id_][property] = [data[id_][property]]
        for j in data[id_][property]:
            groups[j].append(n)
    return groups


def evaluate_property(benchmark_results, property_groups):
    property_results = {}
    for name, group in property_groups.items():
        if len(group) < MIN_SAMPLES:
            continue
        metrics = calculate_metrics(
            [benchmark_results["ground_truths"][i] for i in group],
            [benchmark_results["predictions"][i] for i in group],
        )
        property_results[name] = metrics
    return property_results


# Main
# ====
if __name__ == "__main__":
    # Load the results
    with open(PATH, "r") as f:
        results = json.load(f)
    for benchmark_name, benchmark_results in results.items():
        if benchmark_name == "llm_info":
            continue
        printo(f"\n\nBenchmark: {benchmark_name}")
        printo("=====================================")
        benchmark = get_benchmark_by_name(benchmark_name)

        # Evaluate the overall performance
        metrics = calculate_metrics(
            benchmark_results["ground_truths"], benchmark_results["predictions"]
        )
        printo("Overall performance:")
        printo("--------------------")
        print_metrics(metrics)
        printo("--------------------\n")

        # Evaluate the performance by property
        printo("Performance by property:")
        printo("--------------------")
        for property_name in benchmark.label_tag_groups:
            property_groups = get_groups_by_property(
                benchmark_results["ids"], property_name, benchmark
            )
            property_results = evaluate_property(benchmark_results, property_groups)
            printo(f"{property_name.capitalize()} performance ranking:")
            printo("------------------------------------")
            sorted_results = sorted(
                property_results.items(),
                key=lambda x: (
                    x[1][RANK_BY] if AVERAGE_METHOD else np.mean(x[1][RANK_BY])
                ),
                reverse=True,
            )
            for name, metrics in sorted_results[: 5 if not PRINT_ALL else None]:
                printo(
                    f"{name} ({RANK_BY}; n={len(property_groups[name])}): {metrics[RANK_BY]}"
                )
                if INCLUDE_ALL_METRICS_IN_PROPERTY:
                    print_metrics(metrics)
            if not PRINT_ALL:
                printo("...")
                printo("- 5 Worst performing groups:")
                for name, metrics in sorted_results[-5:][::-1]:
                    printo(
                        f"{name} ({RANK_BY}; n={len(property_groups[name])}): {metrics[RANK_BY]}"
                    )
                    if INCLUDE_ALL_METRICS_IN_PROPERTY:
                        print_metrics(metrics)
            printo("------------------------------------")
    if OUTPUT_FILE:
        OUTPUT_FILE.close()
