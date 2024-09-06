# Benchmark Descriptions

This document outlines the benchmarks used for evaluating question-answering models on various datasets. Each benchmark is designed to assess different aspects of model performance across multiple domains and the Swedish medical context.

## PubMedQA-Swedish-1000

The **PubMedQA-Swedish-1000** benchmark is a Swedish-language version of the PubMedQA dataset, containing 1,000 questions derived from scientific articles' titles and abstracts in the PubMed database. Each question has a yes/no/maybe answer format, aimed at evaluating models' capabilities to understand and reason about medical literature in Swedish.

- **Number of Questions**: 1,000
- **Source**: Translated from the original English PubMedQA dataset.
- **Question Format**: Yes/No/Maybe.
- **Evaluation**: Models are tasked with answering based on scientific abstracts, focusing on comprehension of evidence-based medical information.
- **Significance**: This benchmark tests models' ability to process biomedical texts, crucial for applications such as literature-based knowledge extraction and evidence-based medical practice.

## Swedish Medical Doctors Knowledge Test (SMDT)

The **Swedish Medical Doctors Knowledge Test (SMDT)** consists of 535 multiple-choice questions that cover a wide range of medical specialties. These questions have been adapted from official Swedish medical exams used to test medical students and practitioners.

- **Number of Questions**: 535
- **Scope**: Broad coverage of clinical knowledge across various specialties.
- **Question Format**: Multiple-choice, with five answer options.
- **Significance**: This benchmark assesses broad medical knowledge, ensuring that models can handle the diverse information required in clinical practice.

## Emergency Medicine (SE-EM)

The **Emergency Medicine (SE-EM)** benchmark focuses on scenarios encountered in emergency care, including time-critical medical conditions. These 464 multiple-choice questions simulate emergency department conditions where quick decision-making is crucial.

- **Number of Questions**: 464
- **Content**: Questions cover urgent medical conditions that require swift identification and response.
- **Question Format**: Multiple-choice, with four answer options per question.
- **Significance**: This benchmark evaluates a model’s ability to manage emergency medicine scenarios, ensuring that it can prioritize life-threatening conditions in a timely manner.

## General Medicine (SE-GM)

The **General Medicine (SE-GM)** benchmark evaluates models on common clinical conditions typically seen in general practice. With 666 multiple-choice questions, this benchmark reflects more than 50% of the interactions between general practitioners and patients in primary care.

- **Number of Questions**: 666
- **Content**: Covers over 200 common disorders seen in primary care, including questions on symptoms, diagnosis, and treatment plans.
- **Question Format**: Multiple-choice, with four answer options.
- **Significance**: This benchmark assesses models' ability to accurately diagnose and assess the severity of common disorders in a general medicine context.

## General Practitioner

The **General Practitioner** benchmark mirrors real-world scenarios that general practitioners (GPs) encounter daily. It includes questions on general health, symptoms, diagnosis, and treatment options. The benchmark is designed to test the practical medical knowledge that AI models would need when dealing with a broad range of common clinical issues.

- **Number of Questions**: Varies depending on the case.
- **Content**: Focused on practical, day-to-day medical cases that a GP would handle, from routine checkups to symptom analysis and diagnosis.
- **Significance**: This benchmark is critical for testing a model’s effectiveness in understanding and handling practical medical inquiries, from symptoms to treatment, in a real-world setting.

---

# Adding a New Benchmark

To integrate a new benchmark into our evaluation framework, follow these steps:

### 1. **Prepare the Benchmark**:
   - Create a new directory in the `benchmarks` folder, named after the new benchmark.
   - Add all necessary files, such as datasets and configuration files, to this directory.

### 2. **Update Documentation**:
   - Amend this `BENCHMARK_DESCRIPTIONS.md` file to include a detailed description of the new benchmark, its goals, and content.

### 3. **Update Codebase**:
   - Modify the `run_llm/benchmark.py` file to define the setup of the new benchmark.
   - Add the benchmark class to `benchmark_set_up.py`. Define the specific behaviors for methods like `get_ground_truth()`, `detect_answers()`, and `final_prompt_format()`. Make sure the class adheres to the `Benchmark` abstract base class's requirements.

### 4. **Configure Execution**:
   - Update `run_llm/huggingface.py` and `run_llm/api_models.py` to include instances of the new benchmark in the `BENCHMARKS` list.
   - Adjust system prompts and other parameters as necessary to ensure that the benchmark is tested properly.

### 5. **Submit for Review**:
   - Once all changes are made, open a Pull Request with these modifications. Our team will review the changes and provide feedback.

---

By ensuring consistency and proper documentation, this approach allows for easy integration and evaluation of new benchmarks, fostering a dynamic and transparent process for advancing AI models in the Swedish medical domain.
