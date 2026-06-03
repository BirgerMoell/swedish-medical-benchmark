import json
import argparse
import numpy as np
import torch
import transformers
import benchmark_set_up as benchmarks
import datetime
from pathlib import Path

from functools import lru_cache
from tqdm import tqdm


# Configuration
# =============
MODEL_NAME = "birgermoell/eir"
PubMedQALSWE_SYSTEM_PROMPT = "Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
GeneralPractioner_SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen."
SwedishDoctorsExam = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen. Svara med hela svarsalternativet. Utöver det är det viktigt att du inte inkluderar någon annan text i ditt svar."


def pubmedqa_swe():
    return benchmarks.PubMedQALSWE(
        prompt=PubMedQALSWE_SYSTEM_PROMPT
        + "\n\nFråga:\n{question} svara bara 'ja', 'nej' eller 'kanske'"
    )


def general_practitioner():
    return benchmarks.GeneralPractioner(
        prompt=GeneralPractioner_SYSTEM_PROMPT
        + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."
    )


def emergency_medicine():
    return benchmarks.EmergencyMedicine(
        prompt=GeneralPractioner_SYSTEM_PROMPT
        + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."
    )


def swedish_doctors_exam():
    return benchmarks.SwedishDoctorsExam(
        prompt=SwedishDoctorsExam
        + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."
    )


BENCHMARK_FACTORIES = {
    "PubMedQA-L-SWE": pubmedqa_swe,
    "GeneralPractioner": general_practitioner,
    "EmergencyMedicine": emergency_medicine,
    "SwedishDoctorsExam": swedish_doctors_exam,
}
DEFAULT_BENCHMARKS = ["PubMedQA-L-SWE"]
PIPELINE_PARAMS = {"do_sample": False}


# Functions
# =========
@lru_cache(maxsize=1)
def load_pipeline(
    model=MODEL_NAME,
) -> transformers.pipelines.text_generation.TextGenerationPipeline:
    return transformers.pipeline(
        "text-generation", model=model, torch_dtype=torch.float16, device_map="auto"
    )


def get_response(messages: list[str]) -> str:
    response = messages[0]["generated_text"][-1]
    assert response["role"] == "assistant"
    return response["content"].lower()


def fmt_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def timestamp():
    return datetime.datetime.now().isoformat()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or more SMLB benchmarks with a local Hugging Face model."
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARK_FACTORIES),
        default=DEFAULT_BENCHMARKS,
        help="Benchmarks to run. Keep this list fixed for before/after comparisons.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Where to save predictions and metadata.",
    )
    return parser.parse_args()


# Main
# ====
if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(args.model_name)
    benchmarks_to_run = [BENCHMARK_FACTORIES[name]() for name in args.benchmarks]
    result = {
        "llm_info": {
            "model": args.model_name,
            "benchmarks": args.benchmarks,
            "pipeline_params": PIPELINE_PARAMS,
            "model_run": timestamp(),
            "runner": "run_llm/huggingface.py",
        },
    }
    for benchmark in benchmarks_to_run:
        llm_results = []
        ids = []
        ground_truths = benchmark.get_ground_truth()

        for k, v in tqdm(benchmark.data.items(), desc=f"Processing {benchmark.name}"):
            messages = [fmt_message("user", benchmark.final_prompt_format(v))]
            out = pipeline(
                messages,
                max_new_tokens=benchmark.max_tokens,
                do_sample=PIPELINE_PARAMS["do_sample"],
            )
            llm_results.append(get_response(out))
            predictions = benchmark.detect_answers(llm_results)
            ids.append(k)
            result[benchmark.name] = {
                "prompt": benchmark.prompt,
                "ground_truths": ground_truths.tolist(),
                "predictions": predictions.tolist(),
                "ids": ids,
            }
            with output_path.open("w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        assert len(ground_truths) == len(predictions)

        print(f"Accuracy {(predictions == ground_truths).sum() / len(ground_truths)}")
        print(f"Malformed answers {(predictions == 'missformat').sum()}")

    print(
        f"Done! Results saved to {output_path}. You can now run evaluate_performance.py for detailed metrics."
    )
