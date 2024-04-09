import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import transformers

MODEL_NAME = "birgermoell/eir"
BENCHMARK_PATH = Path("benchmarks/pubmedqa/data/ori_pqal_swe.json")

DATA = json.loads(BENCHMARK_PATH.read_text())


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


def eval(system_prompt: str) -> None:
    pipeline = load_pipeline()

    predictions = []
    for k, v in DATA.items():
        prompt = f"{system_prompt}\n\nFråga\n{v['question']} svara bara 'ja', 'nej' eller 'kanske'"
        messages = [fmt_message("user", prompt)]

        out = pipeline(messages, max_new_tokens=10, do_sample=True, temperature=0)
        answer = get_response(out)

        if "ja" in answer:
            predictions.append("ja")
        elif "nej" in answer:
            predictions.append("nej")
        elif "kanske" in answer:
            predictions.append("kanske")
        else:
            predictions.append("missformat")

    ground_truths = np.asarray([v["final_decision"] for v in DATA.values()])
    predictions = np.asarray(predictions)
    assert len(ground_truths) == len(predictions)

    print(f"Accuracy {(predictions == ground_truths).sum() / len(ground_truths)}")
    print(f"Malformed answers {predictions[predictions == 'missformat'].sum()}")


if __name__ == "__main__":
    system_prompt = "Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
    eval(system_prompt)
