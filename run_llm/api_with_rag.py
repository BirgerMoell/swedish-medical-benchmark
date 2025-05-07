import os
import json
import numpy as np
import sys
import os
import benchmark_set_up as benchmarks
import datetime

from gemini_client import GeminiClient
from rag.rag_system import SwedishRAGSystem
from litellm import completion
from tqdm import tqdm
from time import sleep

# models
gemini_model = "gemini-2.5-flash-preview-04-17"
gpt_model = "gpt-4-0125-preview"

# Configuration
# =============
MODEL_NAME = gemini_model  # Specify the model to use
PubMedQALSWE_SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
GeneralPractioner_SYSTEM_PROMPT = "Du är en utmärkt läkare och specialist i allmänmedicin. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen."
SwedishDoctorsExam = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen. Svara med hela svarsalternativet. Utöver det är det viktigt att du inte inkluderar någon annan text i ditt svar."

BENCHMARKS = [
    # benchmarks.PubMedQALSWE(
    #      prompt=PubMedQALSWE_SYSTEM_PROMPT
    #      + "\n\nFråga:\n{question}\n\nSvara endast 'ja', 'nej' eller 'kanske'."
    #  ),
    # benchmarks.EmergencyMedicine(prompt=GeneralPractioner_SYSTEM_PROMPT + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."),
    benchmarks.SwedishDoctorsExam(prompt=SwedishDoctorsExam + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."),
]

# Initialize RAG system
rag_system = SwedishRAGSystem(
    data_dir="/Users/birgermoell/Documents/polymath/swedish-medical-benchmark/data",
    index_dir="/Users/birgermoell/Documents/polymath/swedish-medical-benchmark/rag/large_rag"
)

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

def get_relevant_context(question: str, k: int = 3) -> str:
    """Get relevant context for a question using RAG."""
    try:
        print(f"Attempting to search with question: {question[:100]}...")  # Print first 100 chars
        results = rag_system.search(question, k=k)
        print(f"Search returned {len(results) if results else 0} results")

        if results:
            context = results[0].page_content  # Only take first result
            print(f"Successfully built context with {len(context)} characters")
            context_to_return = f"\nDen här medicinska informationen kan vara relevant för att svara på frågan:\n{context}\n\nUtgå ifrån din egen bästa bedömning om informationen är hjälpsam:"
            print(f"Context to be returned:\n{context_to_return}")
            return context_to_return
        else:
            print("No results found in search")
    except Exception as e:
        print(f"Warning: Could not retrieve context: {str(e)}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
    return ""

# Main
# ====
if __name__ == "__main__":
    result = {
        "llm_info": {
            "model": MODEL_NAME,
            "model_run": timestamp(),
        },
    }
    
    # Create index if it doesn't exist
    if not os.path.exists("/Users/birgermoell/Documents/polymath/swedish-medical-benchmark/rag/large_rag"):
        print("Creating RAG index...")
        rag_system.create_index()
    
    for benchmark in BENCHMARKS:
        print(f"\nProcessing benchmark: {benchmark.name}")
        print("=" * 50)
        
        llm_results = []
        ids = []
        ground_truths = benchmark.get_ground_truth()
        
        # Create progress bar with additional metrics
        pbar = tqdm(benchmark.data.items(), desc=f"Processing {benchmark.name}")
        
        for k, v in pbar:
            # Get base prompt
            base_prompt = benchmark.final_prompt_format(v)

            
            # Get relevant context using RAG
            context = get_relevant_context(v["QUESTION"])
            
            # Combine context with base prompt
            content = context + "\n" + base_prompt if context else base_prompt
            
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
            
            sleep(3)  # To avoid rate limiting

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
