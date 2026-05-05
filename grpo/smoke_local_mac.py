#!/usr/bin/env python3
"""Local smoke checks for the Swedish medical GRPO dataset.

This is meant for a laptop sanity check. It does not run GRPO training. It
validates the JSONL files, exercises the reward function, optionally checks the
Hugging Face datasets loader, and can tokenize prompts with a locally cached
model.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from train_grpo_cuda import medical_answer_reward


REQUIRED_FIELDS = {
    "id",
    "benchmark",
    "prompt",
    "answer",
    "reward_type",
    "split",
    "source_file",
    "trust_tier",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_splits(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    splits = {}
    for split in ["train", "validation", "test"]:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        splits[split] = read_jsonl(path)
    return splits


def validate_rows(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors = []
    seen_ids = set()
    for split, rows in splits.items():
        if not rows:
            errors.append(f"{split} split is empty")
        for index, row in enumerate(rows):
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                errors.append(f"{split}[{index}] missing fields: {sorted(missing)}")
            if row.get("split") != split:
                errors.append(f"{split}[{index}] has split={row.get('split')!r}")
            row_id = row.get("id")
            if row_id in seen_ids:
                errors.append(f"duplicate id: {row_id}")
            seen_ids.add(row_id)
            if not str(row.get("prompt", "")).strip():
                errors.append(f"{split}[{index}] empty prompt")
            if row.get("reward_type") == "mcq_letter_exact" and row.get("answer") not in {"A", "B", "C", "D", "E"}:
                errors.append(f"{split}[{index}] invalid MCQ answer: {row.get('answer')!r}")
            if row.get("reward_type") == "label_exact" and row.get("answer") not in {"ja", "nej", "kanske"}:
                errors.append(f"{split}[{index}] invalid label answer: {row.get('answer')!r}")
    return errors


def check_rewards(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[str]:
    errors = []
    sample = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    for row in sample:
        gold_completion = row["answer"]
        gold_reward = medical_answer_reward([gold_completion], [row["answer"]], [row["reward_type"]])[0]
        if gold_reward < 1.0:
            errors.append(f"gold answer did not score for {row['id']}: {gold_reward}")

        wrong = "A" if row["answer"] != "A" else "B"
        if row["reward_type"] == "label_exact":
            wrong = "nej" if row["answer"] != "nej" else "ja"
        wrong_reward = medical_answer_reward([wrong], [row["answer"]], [row["reward_type"]])[0]
        if wrong_reward >= 1.0:
            errors.append(f"wrong answer scored too high for {row['id']}: {wrong_reward}")
    return errors


def check_datasets_loader(dataset_dir: Path) -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        return "datasets not installed; skipped"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(dataset_dir / "train.jsonl"),
            "validation": str(dataset_dir / "validation.jsonl"),
            "test": str(dataset_dir / "test.jsonl"),
        },
    )
    return ", ".join(f"{split}={len(dataset[split])}" for split in dataset)


def check_tokenizer(model_name: str, prompts: list[str], allow_downloads: bool) -> str:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "transformers not installed; skipped"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=not allow_downloads)
    lengths = [len(tokenizer(prompt).input_ids) for prompt in prompts]
    return f"{model_name}: min={min(lengths)}, mean={sum(lengths) / len(lengths):.1f}, max={max(lengths)} tokens"


def torch_status() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed"

    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, mps={mps}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/grpo"))
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()

    splits = load_splits(args.dataset_dir)
    rows = [row for split_rows in splits.values() for row in split_rows]
    errors = validate_rows(splits)
    errors.extend(check_rewards(rows, args.sample_size, args.seed))

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"torch: {torch_status()}")
    print("jsonl counts:", ", ".join(f"{split}={len(split_rows)}" for split, split_rows in splits.items()))
    print("benchmarks:", dict(Counter(row["benchmark"] for row in rows)))
    print("reward types:", dict(Counter(row["reward_type"] for row in rows)))
    print(f"datasets loader: {check_datasets_loader(args.dataset_dir)}")

    if args.tokenizer_model:
        prompts = [row["prompt"] for row in random.Random(args.seed).sample(rows, min(args.sample_size, len(rows)))]
        try:
            print(f"tokenizer: {check_tokenizer(args.tokenizer_model, prompts, args.allow_downloads)}")
        except Exception as exc:
            errors.append(f"tokenizer check failed: {exc}")
    else:
        print("tokenizer: skipped; pass --tokenizer-model to check a local/cache model")

    if errors:
        print("\nFAILED")
        for error in errors[:20]:
            print(f"- {error}")
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more")
        sys.exit(1)

    print("\nOK: local GRPO dataset smoke checks passed")


if __name__ == "__main__":
    main()
