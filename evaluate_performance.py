import json
import argparse
import numpy as np

from functools import partial
from pathlib import Path
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
AVERAGE_METHOD = None  # Can be "micro", "macro", "weighted", "samples", None


# Functions
# =========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an SMLB results JSON file."
    )
    parser.add_argument(
        "--results",
        default=PATH,
        help="Path to a results JSON file produced by a runner.",
    )
    parser.add_argument(
        "--save-result-path",
        default=SAVE_RESULT_PATH,
        help="Optional text file where evaluation output should be written.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        default=PRINT_ALL,
        help="Print all property groups instead of only the top and bottom 5.",
    )
    parser.add_argument(
        "--include-all-metrics-in-property",
        action="store_true",
        default=INCLUDE_ALL_METRICS_IN_PROPERTY,
        help="Print all metrics for each property group.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["accuracy", "precision", "recall", "f1"],
        default=RANK_BY,
        help="Metric used to rank property groups.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES,
        help="Minimum samples required for property-level reporting.",
    )
    parser.add_argument(
        "--average-method",
        choices=["micro", "macro", "weighted", "samples"],
        default=AVERAGE_METHOD,
        help="Optional sklearn averaging method for precision, recall, and F1.",
    )
    return parser.parse_args()


def calculate_metrics(
    ground_truths: list[str],
    predictions: list[str],
    average_method: str | None = AVERAGE_METHOD,
):
    metrics = {
        "accuracy": accuracy_score(ground_truths, predictions),
        "precision": precision_score(
            ground_truths, predictions, average=average_method, zero_division=0
        ),
        "recall": recall_score(
            ground_truths, predictions, average=average_method, zero_division=0
        ),
        "f1": f1_score(
            ground_truths, predictions, average=average_method, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(ground_truths, predictions),
    }
    return {
        k: list(v) if isinstance(v, np.ndarray) and k != "confusion_matrix" else v
        for k, v in metrics.items()
    }


def print_metrics(metrics: dict, print_fn=print):
    print_fn(f"Accuracy: {metrics['accuracy']}")
    print_fn(f"Precision: {metrics['precision']}")
    print_fn(f"Recall: {metrics['recall']}")
    print_fn(f"F1: {metrics['f1']}")
    print_fn(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


def get_groups_by_property(ids: list[str], property: str, benchmark: Benchmark):
    groups = defaultdict(list[int])
    data = benchmark.data if not hasattr(benchmark, "og_data") else benchmark.og_data
    for n, id_ in enumerate(ids):
        if not isinstance(data[id_][property], list):
            data[id_][property] = [data[id_][property]]
        for j in data[id_][property]:
            groups[j].append(n)
    return groups


def evaluate_property(
    benchmark_results,
    property_groups,
    average_method: str | None = AVERAGE_METHOD,
    min_samples: int = MIN_SAMPLES,
):
    property_results = {}
    for name, group in property_groups.items():
        if len(group) < min_samples:
            continue
        metrics = calculate_metrics(
            [benchmark_results["ground_truths"][i] for i in group],
            [benchmark_results["predictions"][i] for i in group],
            average_method,
        )
        property_results[name] = metrics
    return property_results


# Main
# ====
def main():
    args = parse_args()
    output_file = None
    if args.save_result_path:
        save_path = Path(args.save_result_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = save_path.open("w")
    printo = partial(print, file=output_file) if output_file else print

    # Load the results
    try:
        with open(args.results, "r") as f:
            results = json.load(f)
        for benchmark_name, benchmark_results in results.items():
            if benchmark_name == "llm_info":
                continue
            printo(f"\n\nBenchmark: {benchmark_name}")
            printo("=====================================")
            benchmark = get_benchmark_by_name(benchmark_name)

            # Evaluate the overall performance
            metrics = calculate_metrics(
                benchmark_results["ground_truths"],
                benchmark_results["predictions"],
                args.average_method,
            )
            printo("Overall performance:")
            printo("--------------------")
            print_metrics(metrics, printo)
            printo("--------------------\n")

            # Evaluate the performance by property
            printo("Performance by property:")
            printo("--------------------")
            for property_name in benchmark.label_tag_groups:
                property_groups = get_groups_by_property(
                    benchmark_results["ids"], property_name, benchmark
                )
                property_results = evaluate_property(
                    benchmark_results,
                    property_groups,
                    args.average_method,
                    args.min_samples,
                )
                printo(f"{property_name.capitalize()} performance ranking:")
                printo("------------------------------------")
                sorted_results = sorted(
                    property_results.items(),
                    key=lambda x: (
                        x[1][args.rank_by]
                        if args.average_method
                        else np.mean(x[1][args.rank_by])
                    ),
                    reverse=True,
                )
                for name, metrics in sorted_results[: 5 if not args.print_all else None]:
                    printo(
                        f"{name} ({args.rank_by}; n={len(property_groups[name])}): {metrics[args.rank_by]}"
                    )
                    if args.include_all_metrics_in_property:
                        print_metrics(metrics, printo)
                if not args.print_all:
                    printo("...")
                    printo("- 5 Worst performing groups:")
                    for name, metrics in sorted_results[-5:][::-1]:
                        printo(
                            f"{name} ({args.rank_by}; n={len(property_groups[name])}): {metrics[args.rank_by]}"
                        )
                        if args.include_all_metrics_in_property:
                            print_metrics(metrics, printo)
                printo("------------------------------------")
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
