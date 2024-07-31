import os
import json
import numpy as np
import benchmark_set_up as benchmarks
import datetime

from litellm import completion
from tqdm import tqdm
from time import sleep


# Configuration
# =============
MODEL_NAME = (
    "gpt-4-0125-preview"  # Specify the model to use. It doesn't need to be from OpenAI.
)
PubMedQALSWE_SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
GeneralPractioner_SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen."
# Make sure to uncomment the benchmarks you want to run
BENCHMARKS = [
    benchmarks.PubMedQALSWE(
        prompt=PubMedQALSWE_SYSTEM_PROMPT
        + "\n\nFråga:\n{question}\n\nSvara endast 'ja', 'nej' eller 'kanske'."
    )
    # Uncomment to also run the GeneralPractioner benchmark
    # benchmarks.GeneralPractioner(
    #     prompt=GeneralPractioner_SYSTEM_PROMPT
    #     + "\n\nFråga:\n{question}\nAlternativ:{options}\n\nSvara endast ett av alternativen."
    # ),
]
os.environ["OPENAI_API_KEY"] = "set-key-here"  # Set your api key and key name.


# Functions
# =========
def get_response(messages: list[str]) -> str:
    response = messages.choices[0].message
    assert response["role"] == "assistant"
    return response["content"].lower()


def timestamp():
    return datetime.datetime.now().isoformat()


# Main
# ====
if __name__ == "__main__":
    result = {
        "llm_info": {
            "model": MODEL_NAME,
            "model_run": timestamp(),
        },
    }
    for benchmark in BENCHMARKS:
        llm_results = []
        ids = []
        ground_truths = benchmark.get_ground_truth()

        for k, v in tqdm(benchmark.data.items(), desc=f"Processing {benchmark.name}"):
            messages = [
                {
                    "role": "user",
                    "content": benchmark.final_prompt_format(v),
                }
            ]
            out = completion(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=10,
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
            with open("./results.json", "w") as f:
                json.dump(result, f)
            sleep(0)  # To avoid rate limiting, change as needed.

        assert len(ground_truths) == len(predictions)

        print(f"Accuracy {(predictions == ground_truths).sum() / len(ground_truths)}")
        print(f"Malformed answers {(predictions == 'missformat').sum()}")

    print(
        "Done! You can now run the evaluate_results.py script to get detailed performance metrics."
    )
