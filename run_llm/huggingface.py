import json
import numpy as np
import torch
import transformers
import benchmark_set_up as benchmarks

from functools import lru_cache


# Configuration
# =============
MODEL_NAME = "birgermoell/eir"
PubMedQALSWE_SYSTEM_PROMPT = "Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
BENCHMARKS = [
    benchmarks.PubMedQALSWE(
        prompt=PubMedQALSWE_SYSTEM_PROMPT
        + "\n\nFråga\n{question} svara bara 'ja', 'nej' eller 'kanske'"
    )
]
PIPELINE_PARAMS = {"max_new_tokens": 10, "do_sample": False}


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


# Main
# ====
if __name__ == "__main__":
    pipeline = load_pipeline()
    result = {
        "llm_info": {"model": MODEL_NAME, "pipeline_params": PIPELINE_PARAMS},
    }
    for benchmark in BENCHMARKS:
        llm_results = []
        ids = []
        for k, v in benchmark.data.items():
            messages = [
                fmt_message("user", benchmark.prompt.format(question=v["QUESTION"]))
            ]
            out = pipeline(
                messages,
                max_new_tokens=PIPELINE_PARAMS["max_new_tokens"],
                do_sample=PIPELINE_PARAMS["do_sample"],
            )
            llm_results.append(get_response(out))
            ids.append(k)

        ground_truths = benchmark.get_ground_truth()
        predictions = benchmark.detect_answers(llm_results)
        assert len(ground_truths) == len(predictions)
        result[benchmark.name] = {
            "prompt": benchmark.prompt,
            "ground_truths": ground_truths.tolist(),
            "predictions": predictions.tolist(),
            "ids": ids,
        }

        print(f"Accuracy {(predictions == ground_truths).sum() / len(ground_truths)}")
        print(f"Malformed answers {predictions[predictions == 'missformat'].sum()}")

    print(
        "Done! You can now run the evaluate_results.py script to get detailed performance metrics."
    )
    with open("./results.json", "w") as f:
        json.dump(result, f)
