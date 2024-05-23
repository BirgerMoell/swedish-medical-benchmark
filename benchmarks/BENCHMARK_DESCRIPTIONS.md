# Benchmark descriptions

## PubMedQA-L-SWE
The PubMedQA-L-SWE benchmark is a Swedish language version of the PubMedQA benchmark. The benchmark consists of 1,000 questions 
and answers from the PubMedQA dataset, translated into Swedish. The questions are based on the titles and abstracts of 
scientific articles from the PubMed database. The benchmark is designed to evaluate the performance of question answering models
on scientific text in Swedish.

## General Practioner
Description to be added.

# Add a new benchmark
To add a new benchmark, follow these steps:
* Create a new directory in the `benchmarks` folder with the name of the benchmark and add all the necessary files there.
* Update the `BENCHMARK_DESCRIPTIONS.md` file with a description of the new benchmark.
* Update the `run_llm/huggingface.py` and `run_llm/api_models.py` file to include the new benchmark in the evaluation process.
* Your done! Open a PR with the changes and we will review it as soon as possible.
