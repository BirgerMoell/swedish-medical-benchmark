import os
import json
import numpy as np
import benchmark_set_up as benchmarks
import datetime
from gemini_client import GeminiClient

from litellm import completion
from tqdm import tqdm
from time import sleep

# models
gemini_model = "gemini-2.5-flash-preview-04-17"
gpt_model = "gpt-4-0125-preview"

# Configuration
# =============
MODEL_NAME = (
    gemini_model  # Specify the model to use. It doesn't need to be from OpenAI.
)
PubMedQALSWE_SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
GeneralPractioner_SYSTEM_PROMPT = "Du är en utmärkt läkare och specialist i allmänmedicin. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen."
SwedishDoctorsExam = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen. Svara med hela svarsalternativet. Utöver det är det viktigt att du inte inkluderar någon annan text i ditt svar."

# Make sure to uncomment the benchmarks you want to run
BENCHMARKS = [
    benchmarks.GeneralPractioner(prompt=GeneralPractioner_SYSTEM_PROMPT + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."),
]

# get the api key from dotenv
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
gemini_client = GeminiClient()

# Functions
# =========
def get_response(messages: list[str]) -> str:
    if MODEL_NAME.startswith("gemini") and True:
        # Handle Gemini response format
        return messages.lower()
    else:
        # Handle other models (like GPT) response format
        response = messages.choices[0].message
        assert response["role"] == "assistant"
        return response["content"].lower()

def timestamp():
    return datetime.datetime.now().isoformat()

def calculate_metrics(predictions, ground_truths):
    """Calculate and return accuracy metrics."""
    correct = (predictions == ground_truths).sum()
    total = len(ground_truths)
    accuracy = correct / total if total > 0 else 0
    malformed = (predictions == 'missformat').sum()
    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'malformed': malformed
    }

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
        print(f"\nProcessing benchmark: {benchmark.name}")
        print("=" * 50)
        
        llm_results = []
        ids = []
        ground_truths = benchmark.get_ground_truth()
        
        # Create progress bar with additional metrics
        pbar = tqdm(benchmark.data.items(), desc=f"Processing {benchmark.name}")
        
        for k, v in pbar:
            content = benchmark.final_prompt_format(v)
            
            if MODEL_NAME.startswith("gemini"):
                # Use Gemini client for Gemini models
                out = gemini_client.generate_content(content)
            else:
                # Use litellm for other models
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                out = completion(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=benchmark.max_tokens,
                )
            
            response = get_response(out)
            print(f"Response: {response}")

            llm_results.append(response)
            
            predictions = benchmark.detect_answers(llm_results)
            ids.append(k)
            
            # Calculate current metrics
            metrics = calculate_metrics(predictions, ground_truths[:len(predictions)])
            
            # Update progress bar description with metrics
            pbar.set_postfix({
                'Correct': metrics['correct'],
                'Total': metrics['total'],
                'Accuracy': f"{metrics['accuracy']:.2%}",
                'Malformed': metrics['malformed']
            })
            
            # Save results after each prediction
            result[benchmark.name] = {
                "prompt": benchmark.prompt,
                "ground_truths": ground_truths.tolist(),
                "predictions": predictions.tolist(),
                "ids": ids,
            }
            with open("./results.json", "w") as f:
                json.dump(result, f)
            
            sleep(3)  # To avoid rate limiting, change as needed.

        # Final metrics
        final_metrics = calculate_metrics(predictions, ground_truths)
        print("\nFinal Results:")
        print(f"Total Questions: {final_metrics['total']}")
        print(f"Correct Answers: {final_metrics['correct']}")
        print(f"Accuracy: {final_metrics['accuracy']:.2%}")
        print(f"Malformed Answers: {final_metrics['malformed']}")

    print(
        "\nDone! You can now run the evaluate_results.py script to get detailed performance metrics."
    )
