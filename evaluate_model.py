import json

import torch
import transformers
from transformers import AutoTokenizer

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

    n = 0
    correct = 0
    malformed_answers = 0

    for key, value in data.items():
        #print(f"ID: {key}")
        #print(f"ID: {key}")
        #print(f"QUESTION: {value['QUESTION']}")

             # print("CONTEXTS:")
        context_summed = ""
        for context in value['CONTEXTS']:
            #print(f"- {context}")
            context_summed += context + " "

        question = value["QUESTION"]

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + " " + system_prompt},
        ]

        print("the message is: ", messages)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = pipeline(
            prompt,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            top_k=50,
            top_p=0.95,
        )
        #print(outputs[0]["generated_text"])
        answer = outputs[0]["generated_text"]
        # strip the answer from the syste
        # split the string on [/INST]
        answer = answer.split("[/INST]")[1]
        # make answer lowercase
        answer = answer.lower()
        print("AI ANSWER: ", answer)

        # check if answer containts ja, nej or kanske
        if "ja" in answer:
            answer = "ja"
        elif "nej" in answer:
            answer = "nej"
        elif "kanske" in answer:
            answer = "kanske"
        else:
            answer = "felaktigt svar"

        print("------AI ANSWER STRUCTURED: ", answer)
        print(f"-----CORRECT ANSWER: {value['final_decision']}\n")

        # save the answer and final decision in the data to a csv file
        value["answer"] = answer
        value["final_decision"] = value["final_decision"]

        if answer == value["final_decision"]:
            correct += 1

        if answer == "felaktigt svar":
            malformed_answers += 1

        # save the data to a csv file that stores the answers and final decisions
        n += 1
    result = correct / n
    print("Percentage of correct answers: ", result)
    print("Number of correct answers: ", correct)
    print("Number of incorrect answers: ", n-correct)
    print("Number of malformed answered: ", malformed_answers)
    # write the results to a csv file

      
system_prompt = "Tänk noggrant steg för steg och ta ett djupt andetag och svara på denna medicinska fråga. Var medveten om att alla svarsalternativ, ja, nej och kanske kan vara möjliga. Svara enbart med följande svarsalternativ: ja, nej, kanske."

system_prompt = "Var vänlig och överväg varje aspekt av föregående medicinska fråga noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."


evaluate_model(data, model, system_prompt)
