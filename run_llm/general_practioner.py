import os
import json
from tqdm import tqdm
import openai

# Configuration
# =============
MODEL_NAME = "gpt-4o"  # Specify the model to use. It doesn't need to be from OpenAI.
SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen."

# Functions
# =========
import os

# use pythondotenv to load the environment variables
from dotenv import load_dotenv
load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPEN_AI_API_KEY)

def get_text_from_open_ai(prompt, system_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )
        text = response.choices[0].message.content
        return text
    except Exception as e:
        print(f"Error in generating text: {e}")
        return None

# Load the questions from the JSON file
with open('/home/bmoell/medical/swedish-medical-benchmark/benchmarks/pubmedqa/data/general_practioner.json', 'r', encoding='utf-8') as file:
    questions = json.load(file)

# Main
# ====
if __name__ == "__main__":
    result = {
        "llm_info": {
            "model": MODEL_NAME,
        },
    }

    llm_results = []
    ids = []
    ground_truths = []
    predictions = []

    for idx, question in enumerate(tqdm(questions, desc="Processing questions")):
        prompt = SYSTEM_PROMPT + f"\n\nFråga:\n{question['question']}\nAlternativ: {', '.join(question['options'])}\n\nSvara endast ett av alternativen."
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        llm_answer = get_text_from_open_ai(prompt, SYSTEM_PROMPT)
        llm_results.append({
            "question": question['question'],
            "llm_answer": llm_answer,
            "correct_answer": question['correct_answer'],
            "is_correct": llm_answer == question['correct_answer'],
        })
        ground_truths.append(question['correct_answer'])
        predictions.append(llm_answer)
        ids.append(idx)

    result["questions"] = llm_results

    with open("./results.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    correct_count = sum(1 for result in llm_results if result["is_correct"])
    total_questions = len(questions)
    accuracy = correct_count / total_questions * 100

    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")