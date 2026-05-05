# Swedish Medical GRPO

This directory turns trusted parts of the Swedish Medical Benchmark into a small
RLVR/GRPO dataset and provides a CUDA training script for TRL.

The default build excludes translated MedQA. It also excludes translated
PubMedQA unless you opt in, because the cleanest first experiment is native
Swedish medical multiple-choice data with deterministic rewards.

## Build the dataset

From the repository root:

```bash
python grpo/build_grpo_dataset.py --output-dir data/grpo
```

Optional translated PubMedQA:

```bash
python grpo/build_grpo_dataset.py --output-dir data/grpo --include-translated-pubmedqa
```

The output files are:

- `data/grpo/train.jsonl`
- `data/grpo/validation.jsonl`
- `data/grpo/test.jsonl`
- `data/grpo/manifest.json`

Each row includes a Swedish prompt, the compact gold answer, source provenance,
reward type, benchmark name, split, and trust tier.

## Install training dependencies

Use a CUDA environment for training:

```bash
python -m venv .venv-grpo
source .venv-grpo/bin/activate
pip install -r requirements-grpo.txt
```

If your CUDA/PyTorch setup needs a specific wheel index, install the matching
`torch` build first, then install the rest of the file.

## Sanity check locally on a Mac

The Mac smoke script validates the JSONL splits and reward function without
running GRPO training:

```bash
python grpo/smoke_local_mac.py --dataset-dir data/grpo
```

If `datasets` is installed, it also checks the Hugging Face JSON loader. To
check prompt token lengths against a locally cached model:

```bash
python grpo/smoke_local_mac.py \
  --dataset-dir data/grpo \
  --tokenizer-model google/gemma-2-2b-it
```

Add `--allow-downloads` only if you want Hugging Face to fetch the tokenizer.

## Run GRPO on CUDA

Example for a small instruction model:

```bash
python grpo/train_grpo_cuda.py \
  --model-name google/gemma-2-2b-it \
  --dataset-dir data/grpo \
  --output-dir outputs/gemma2-2b-smlb-grpo \
  --max-steps 200 \
  --num-generations 4 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --bf16
```

For a smaller GPU, reduce `--num-generations`, `--per-device-train-batch-size`,
and `--max-prompt-length`. LoRA is enabled by default with `--lora-r 16`; set
`--lora-r 0` for full fine-tuning if you have enough VRAM.

## What the reward does

`train_grpo_cuda.py` uses a deterministic reward:

- `mcq_letter_exact`: reward the correct first answer letter, `A` through `E`.
- `label_exact`: reward the correct compact label, currently `ja`, `nej`, or
  `kanske` for optional PubMedQA.
- A small format bonus rewards answers that contain only the compact answer.

This is intended for medical education and reasoning experiments. It is not a
clinical deployment recipe.
