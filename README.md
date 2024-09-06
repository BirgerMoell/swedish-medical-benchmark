# 🇸🇪 Swedish Medical Benchmark 🏥💻

<img src="logo.png">
Welcome to the official repository for the Swedish Medical Benchmark! This project aims to revolutionize how we assess and develop AI models in the medical domain, specifically tailored for the Swedish language. With your help, we can create a more inclusive, accurate, and impactful AI in healthcare. Let's make AI work for everyone!

## Goals 🎯

This project focuses on three primary goals:

## Translate Benchmarks to Swedish 📚➡️🇸🇪

Making existing benchmarks accessible to the Swedish-speaking medical community is crucial. This step involves:

Identifying key medical benchmarks in AI.
Translating these benchmarks into Swedish.
Ensuring the translations maintain the clinical integrity of the original benchmarks.
Create New Benchmark for Swedish 🛠️🆕

## Benchmarks

## 🚀 Benchmarks

We use multiple datasets to evaluate AI models, including:

| **Benchmark**                  | **Questions** | **Description**                                                                                        |
|---------------------------------|---------------|--------------------------------------------------------------------------------------------------------|
| **PubMedQA-Swedish**            | 1000          | Translated PubMedQA questions with yes/no/maybe answers; tests models’ comprehension of medical literature. |
| **Medical Doctors Knowledge Test** | 535          | Adapted from Swedish clinical exams; assesses broad medical knowledge.                                   |
| **Emergency Medicine (SE-EM)**  | 464           | Time-critical scenarios for emergency medicine.                                                         |
| **General Medicine (SE-GM)**    | 666           | Covers 200+ common disorders in general medicine; evaluates diagnosis and assessment.                    |


See more information regarding implemented benchamrks in the [Benchmarks readme](benchmarks/BENCHMARK_DESCRIPTIONS.md) file.

## Develop benchmarks specifically for the Swedish context, incorporating

- Unique medical terminology and practices in Sweden.
- Diverse datasets representing Swedish demographics.
- Collaboration with Swedish medical professionals to ensure relevance and accuracy.

## Preliminary Results of LLMs on the Swedish Medical Benchmark

| **Model**      | **PQ-S** | **SMDT** | **EM** | **GM**  | **SMB** |
|----------------|----------|----------|--------|---------|---------|
| **GPT-4o**     | -        | **83.18%** | 90.51% | 88.88%  | -       |
| **GPT-4**      | 53.90%   | 79.07%   | **93.10%** | **93.09%** | **75.57%** |
| **Claude-3.5** | -        | **83.74%** | -      | -       | -       |
| **Llama3-70b** | 56.00%   | 69.91%   | 74.35% | 67.57%  | 64.88%  |
| **Llama3-8b**  | 50.50%   | 41.68%   | -      | -       | -       |
| **Llama3.1-70b** | -      | 71.40%   | 62.93% | 71.02%  | -       |
| **Llama3.1-8b** | -       | 6.36%    | -      | -       | -       |
| **Gemma2-9b**  | -        | 61.31%   | -      | -       | -       |
| **Gemma-7b**   | 48.70%   | 27.48%   | -      | -       | -       |
| **EIR**        | 46.50%   | -        | -      | -       | -       |
| **GPT-3.5**    | 27.40%   | -        | -      | -       | -       |

> **Note**: "-" indicates no evaluation. Accuracy is reported as a percentage (%).  
> **PQ-S**: PubMedQA-Swedish-1000  
> **SMDT**: Swedish Medical Doctors Test  
> **EM**: Emergency Medicine  
> **GM**: General Medicine  
> **SMB**: Swedish Medical Benchmark

## Evaluating AI models on these benchmarks to understand their effectiveness and areas for improvement

Implementing a standardized evaluation framework.
Encouraging the submission of AI models for testing.
Publishing results to foster transparency and continuous improvement.
Contributing 🤝
Your expertise and enthusiasm can drive this project forward. Here's how you can contribute:

- Translators 📝: Help us bring existing benchmarks to Swedish speakers.
- Data Scientists and Developers 💻: Work on creating the new benchmark, implementing the evaluation framework, and testing AI models.
- Medical Professionals 🩺: Provide insights into Swedish medical practices and validate the clinical relevance of benchmarks.

## Get Started 🚀

Fork this repository to your account.
Pick a task from the issues tab that resonates with your skills and interests.
Follow the contribution guidelines in the CONTRIBUTING.md file for detailed instructions on how to make your contributions count.
Stay Connected 💬
Join our community on Discord for discussions, updates, and collaboration opportunities. Together, we can make a difference in healthcare AI!
<https://discord.gg/AgDx34t2>

## Usage 🛠

> Note: Make sure that you have Python 3.10 or higher installed on your machine.

First  you n️eed to install the requirements:
```bash
pip install -r requirements.txt
```

Then you can run the file associated with the LLM model, make sure to adjust the configuration in the file to your needs. For instance:
```bash
python run_llm/huggingface.py
```

For more detailed metrics run the evaluation script:
```bash
python evaluate_performance.py
```

> Note: The scripts have to be run from the root directory of the project.

## Test files
We have added test files that are encrypted. If you need access please ask the repo maintainers for the password.

To decrypt them. First intall gpg

## Linux
1. Install GPG
If you don't have gpg installed, you can install it:

## Mac
sudo apt-get install gnupg
brew install gnupg
## Windows
Download https://gpg4win.org/

```
gpg your_file.json.gpg
```

### Encrypting new files
If you want to encrypt new files
gpg -c your_file.json

