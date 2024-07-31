# Benchmark Descriptions

This document outlines the benchmarks used for evaluating question answering models on various datasets. Each benchmark is designed to assess different aspects of model performance across multiple domains and languages.

## PubMedQA-L-SWE
The **PubMedQA-L-SWE** benchmark is a Swedish-language version of the PubMedQA benchmark, comprising 1,000 questions and answers translated from the original PubMedQA dataset. These questions are derived from the titles and abstracts of scientific articles in the PubMed database. This benchmark specifically evaluates the effectiveness of question answering models on scientific texts in Swedish.

## General Practitioner
The **General Practitioner** benchmark focuses on common medical questions typical of those encountered in a general practitioner's office. This benchmark includes diverse questions regarding general health, symptoms, diagnoses, and treatments, adapted to reflect real-world clinical situations. It aims to assess the model's capability to handle practical medical inquiries effectively.

# Adding a New Benchmark

To integrate a new benchmark into our evaluation framework, follow these steps:

1. **Prepare the Benchmark**:
   - Create a new directory within the `benchmarks` folder, naming it after the new benchmark.
   - Add all necessary files, such as datasets and configuration files, to this directory.

2. **Update Documentation**:
   - Amend the `BENCHMARK_DESCRIPTIONS.md` file to include a detailed description of the new benchmark.

3. **Update Codebase**:
   - Modify the `run_llm/benchmark.py` file to define the setup of the new benchmark.
   - Add the benchmark class to `benchmark_set_up.py`. Define the specific behaviors for methods like `get_ground_truth()`, `detect_answers()`, and `final_prompt_format()`. Make sure the class adheres to the `Benchmark` abstract base class's requirements.

4. **Configure Execution**:
   - Update `run_llm/huggingface.py` and `run_llm/api_models.py` to include instances of the new benchmark in the `BENCHMARKS` list in .
   - Adjust system prompts and other parameters as necessary to ensure that the benchmark is tested properly.

5. **Submit for Review**:
   - Once all changes are made, open a Pull Request with these modifications. We will review the changes as soon as possible.
