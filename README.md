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

### Pubmedqa

Pubmedqa is a large benchmark of medical questions with yes / no / maybe answers that can be benchmarked with the help of LLMs.
<https://github.com/pubmedqa/pubmedqa>

### Pubmedqa-swe-tiny

We have started translating a  subset of the questions (N=100) for a first benchmark in Swedish.

## Develop benchmarks specifically for the Swedish context, incorporating

- Unique medical terminology and practices in Sweden.
- Diverse datasets representing Swedish demographics.
- Collaboration with Swedish medical professionals to ensure relevance and accuracy.

## Compare Model Performance on the Benchmark 📊🔍

| Metric                     | Eir                | Swe-PubMedQA-100   |
|----------------------------|--------------------|--------------------|
| Total Questions            | 100 📋             | 100 📋             |
| Correct Answers            | 50 ✅              | -                  |
| Incorrect Answers          | 50 ❌              | -                  |
| Malformed Answers          | 0 🚫               | -                  |
| Accuracy                   | 50% 🎯             | -                  |
| Number of yes              | 56 ✔️              | 60 ✔️              |
| Number of no               | 29 ❎              | 30 ❎              |
| Number of maybe            | 15 ➖              | 10 ➖              |

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
python run_llm/huggingface.py.py
```

For more detailed metrics run the evaluation script:
```bash
python evaluate_performance.py
```

> Note: The scripts have to be run from the root directory of the project.
