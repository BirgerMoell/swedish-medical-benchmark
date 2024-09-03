import json

def convert_to_clinical_case(emergency_data):
    clinical_case_data = {}
    case_number = 1

    for case in emergency_data:
        case_description = case["case_description"]
        for question in case["questions"]:
            # Combine the case description with the question text
            full_question = f"{case_description}\n\nQuestion: {question['question']}\nOptions: {', '.join(question['options'])}"
            
            clinical_case_data[str(case_number)] = {
                "Question": full_question,
                "Answer": question["correct_answer"]
            }
            case_number += 1

    return clinical_case_data

def main():
    # Load the emergency_medicine_corrected.json
    with open('/home/bmoell/medical/swedish-medical-benchmark/benchmarks/specialist_questions/gp/fall_descriptions.json', 'r') as f:
        emergency_data = json.load(f)

    # Convert the data
    clinical_case_data = convert_to_clinical_case(emergency_data)

    # Save the result to a new JSON file
    with open('/home/bmoell/medical/swedish-medical-benchmark/benchmarks/specialist_questions/gp/fall_description_clinical_format.json', 'w') as f:
        json.dump(clinical_case_data, f, indent=4, ensure_ascii=False)

    print("Conversion completed successfully. The result is saved as 'clinical_case_converted.json'.")

if __name__ == "__main__":
    main()