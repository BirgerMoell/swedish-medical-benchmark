# Swedish Medical Benchmark (SMLB)

![Swedish Medical Benchmark overview](SMLB.png)

**A framework for evaluating large language models in the Swedish medical domain.**

Created by **Birger Moëll**, **Fabian Farestam**, and **Jonas Beskow**.

SMLB brings together Swedish medical exams, clinical case questions, emergency
medicine scenarios, general medicine cases, and translated biomedical literature
questions into one open benchmark suite. The goal is simple: make it easier to
measure how well language models handle Swedish medical reasoning, terminology,
and clinically relevant multiple-choice tasks.

[Paper](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1557920/full) ·
[PDF](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1557920/pdf) ·
[Benchmark descriptions](benchmarks/BENCHMARK_DESCRIPTIONS.md) ·
[GRPO dataset](grpo/README.md)

## Figures From The Paper

The published paper includes several useful figures for understanding model
performance, PubMedQA-Swedish behavior, and the relationship between SMLB and
general benchmark performance.

| Model performance | PubMedQA-Swedish confusion matrices |
|---|---|
| ![Figure 1: Model performance with 95% confidence intervals](paper-figures/figure-1-model-performance.jpg) | ![Figure 2: Comparison of PubMedQA-Swedish confusion matrices for GPT-4-t and o3](paper-figures/figure-2-pqs-confusion-matrices.jpg) |

| PubMedQA-Swedish F1 metrics | SMDT vs MMLU |
|---|---|
| ![Figure 3: Performance metrics of models on PubMedQA-Swedish](paper-figures/figure-3-pubmedqa-f1.jpg) | ![Figure 4: Comparison of SMDT and MMLU scores across models](paper-figures/figure-4-smdt-vs-mmlu.jpg) |

Figures are reproduced from Moëll, Farestam and Beskow (2025), published in
Frontiers in Artificial Intelligence under the Creative Commons Attribution
License (CC BY).

## What Is Included

| Benchmark | Short name | Questions | Task shape | Notes |
|---|---:|---:|---|---|
| PubMedQA-Swedish | PQ-S | 1,000 | yes/no/maybe | Swedish translation of PubMedQA for medical literature comprehension |
| Swedish Medical Doctors Test | SMDT | 535 | multiple choice | Swedish clinical exam-style questions across broad medical knowledge |
| Emergency Medicine | SE-EM | 464 | multiple choice | Time-critical emergency medicine scenarios |
| General Medicine | SE-GM | 666 | multiple choice | Primary-care-oriented cases covering 200+ common disorders |
| Specialist questions | SMB | varies | multiple choice | Specialty-specific Swedish medical questions |
| GRPO/RLVR dataset | SMLB-GRPO | 2,261 | reward-verifiable MCQ | Native Swedish subset for RL experiments with deterministic rewards |

## Why This Exists

Most medical LLM benchmarks are English-first. Swedish healthcare has its own
clinical language, exam traditions, documentation style, abbreviations, and
practice context. SMLB helps researchers and builders evaluate whether a model
can work with those Swedish-specific conditions instead of only general English
medical knowledge.

This benchmark is intended for research, evaluation, and medical education
experiments. It is not a clinical decision system.

## Quickstart

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
python run_llm/huggingface.py
python evaluate_performance.py
```

Run commands from the repository root.

## GRPO / RLVR Experiments

The repo includes a compact GRPO/RLVR dataset for experiments with verifiable
medical multiple-choice rewards. The default build uses native Swedish sources
and excludes translated MedQA.

```bash
python grpo/build_grpo_dataset.py --output-dir data/grpo
python grpo/smoke_local_mac.py --dataset-dir data/grpo
```

CUDA training with TRL:

```bash
pip install -r requirements-grpo.txt
python grpo/train_grpo_cuda.py \
  --model-name google/gemma-2-2b-it \
  --dataset-dir data/grpo \
  --output-dir outputs/gemma2-2b-smlb-grpo \
  --bf16
```

The GRPO dataset currently contains:

| Split | Rows |
|---|---:|
| train | 1,822 |
| validation | 227 |
| test | 212 |

See [grpo/README.md](grpo/README.md) for details on reward functions, local Mac
smoke checks, optional translated PubMedQA inclusion, and CUDA settings.

## Preliminary Results

Accuracy is reported as a percentage. A dash means that the model has not been
evaluated on that benchmark in the current results table.

| Model | PQ-S | SMDT | EM | GM | SMB |
|---|---:|---:|---:|---:|---:|
| GPT-4o | - | **83.18** | 90.51 | 88.88 | - |
| GPT-4 | 53.90 | 79.07 | **93.10** | **93.09** | **75.57** |
| Claude-3.5 | - | **83.74** | - | - | - |
| Llama3-70b | 56.00 | 69.91 | 74.35 | 67.57 | 64.88 |
| Llama3-8b | 50.50 | 41.68 | - | - | - |
| Llama3.1-70b | - | 71.40 | 62.93 | 71.02 | - |
| Llama3.1-8b | - | 6.36 | - | - | - |
| Gemma2-9b | - | 61.31 | - | - | - |
| Gemma-7b | 48.70 | 27.48 | - | - | - |
| EIR | 46.50 | - | - | - | - |
| GPT-3.5 | 27.40 | - | - | - | - |

## Repository Map

| Path | Purpose |
|---|---|
| [benchmarks/](benchmarks/) | Source benchmark datasets and benchmark descriptions |
| [run_llm/](run_llm/) | Model runner setup for API and Hugging Face models |
| [results/](results/) | Stored model result files |
| [grpo/](grpo/) | GRPO dataset builder, local smoke tests, and CUDA training script |
| [data/grpo/](data/grpo/) | Generated GRPO JSONL train/validation/test splits |
| [docs/](docs/) | GitHub Pages-style project page |
| [website/](website/) | Static website copy |

## For AI Agents

This section is intentionally explicit so coding agents can work on the repo
without rediscovering the basics.

- Project goal: evaluate LLMs on Swedish medical benchmark tasks.
- Default working directory: repository root.
- Use Python 3.10+.
- Main evaluation entry point: `python evaluate_performance.py`.
- Hugging Face runner example: `python run_llm/huggingface.py`.
- Benchmark implementations live in `run_llm/benchmark_set_up.py`.
- Human-readable benchmark descriptions live in `benchmarks/BENCHMARK_DESCRIPTIONS.md`.
- GRPO dataset builder: `python grpo/build_grpo_dataset.py --output-dir data/grpo`.
- Local GRPO sanity check: `python grpo/smoke_local_mac.py --dataset-dir data/grpo`.
- CUDA GRPO script: `python grpo/train_grpo_cuda.py --model-name google/gemma-2-2b-it --dataset-dir data/grpo --bf16`.
- Do not treat SMLB or SMLB-GRPO as clinical deployment data. Use it for evaluation, research, and medical education experiments.
- Translated MedQA is not part of the default GRPO build. PubMedQA-Swedish can be included with `--include-translated-pubmedqa`.
- Preserve provenance fields when adding data. New benchmark rows should keep source file, task type, answer format, and licensing notes clear.

## Contributing

Useful contributions include:

- improving Swedish medical translations and terminology,
- adding clinically reviewed Swedish questions,
- expanding model evaluations,
- improving benchmark loaders and metrics,
- strengthening GRPO/RLVR data quality,
- documenting dataset provenance and limitations.

Medical professionals, translators, data scientists, and developers are all
welcome. Join the community on Discord:
<https://discord.gg/AgDx34t2>

## Test Files

Some test files are encrypted. Ask the maintainers for access if you need them.

Install GPG:

```bash
brew install gnupg
```

Decrypt a file:

```bash
gpg your_file.json.gpg
```

Encrypt a new file:

```bash
gpg -c your_file.json
```

## Citation

```bibtex
@article{moell2025swedish,
  title        = {Swedish Medical LLM Benchmark (SMLB): Development and Evaluation of a Framework for Assessing Large Language Models in the Swedish Medical Domain},
  author       = {Mo{\"e}ll, Birger and Farestam, Fabian and Beskow, Jonas},
  journal      = {Frontiers in Artificial Intelligence},
  volume       = {8},
  pages        = {1557920},
  year         = {2025},
  publisher    = {Frontiers}
}
```
