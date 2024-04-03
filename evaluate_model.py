from transformers import AutoTokenizer
import transformers
import torch
import json

model = "birgermoell/eir"
json_benchmark_file = "benchmarks/pubmedqa/data/ori_pqal_swe.json"

def load_json_file(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

data = load_json_file(json_benchmark_file)

def evaluate_model(data, model, system_prompt):
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for key, value in data.items():
        print(f"ID: {key}")
        print(f"ID: {key}")
        print(f"QUESTION: {value['QUESTION']}")
        question = value["QUESTION"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        print(outputs[0]["generated_text"])
        answer = outputs[0]["generated_text"]
        print("ANSWER: ", answer)
        # print("CONTEXTS:")
        # for context in value['CONTEXTS']:
        #     print(f"- {context}")
        print(f"FINAL DECISION: {value['final_decision']}\n")

system_prompt = "Svara på följande medicinska fråga med en av följande svarsalternativ: Ja, Nej, Kanske"

evaluate_model(data, model)
