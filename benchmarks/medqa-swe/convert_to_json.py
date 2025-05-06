import pandas as pd
import json
from pathlib import Path

def convert_medqa_to_json():
    # Read the CSV file
    df = pd.read_csv('dataset.csv')
    
    # Create a list to store the questions
    questions = []
    
    # Process each row
    for _, row in df.iterrows():
        # Parse the options string into a list
        options = row['options'].split('\n')
        options = [opt.strip() for opt in options if opt.strip()]
        
        # Create the question object
        question = {
            'question': row['question'],
            'options': options,
            'answer': row['answer'],
            'date': row['date'],
            'part': row['part']
        }
        questions.append(question)
    
    # Create the final JSON structure
    output = {
        'name': 'MedQA Swedish',
        'description': 'Swedish medical questions dataset',
        'questions': questions
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path('benchmarks/medqa-swe')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to JSON file
    output_file = output_dir / 'medqa_swe.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Output written to {output_file}")

if __name__ == "__main__":
    convert_medqa_to_json()