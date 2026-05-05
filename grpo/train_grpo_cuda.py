#!/usr/bin/env python3
"""Run GRPO on the Swedish medical GRPO dataset with a CUDA GPU.

This script is intentionally small and hackable. It is meant for experiments on
models such as google/gemma-2-2b-it, not for clinical deployment.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


MCQ_RE = re.compile(r"^\s*([A-Ea-e])\b|^\s*([A-Ea-e])\s*[\).:]")
LABEL_RE = re.compile(r"\b(ja|nej|kanske|yes|no|maybe)\b", re.I)
LABEL_MAP = {"yes": "ja", "no": "nej", "maybe": "kanske", "ja": "ja", "nej": "nej", "kanske": "kanske"}


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def extract_prediction(text: str, reward_type: str) -> str | None:
    stripped = text.strip()
    if reward_type == "mcq_letter_exact":
        match = MCQ_RE.search(stripped)
        return (match.group(1) or match.group(2)).upper() if match else None
    if reward_type == "label_exact":
        match = LABEL_RE.search(stripped)
        return LABEL_MAP.get(match.group(1).lower()) if match else None
    return None


def medical_answer_reward(completions: list[Any], answer: list[str], reward_type: list[str], **_: Any) -> list[float]:
    """Reward clean exact answers.

    Returns 1.0 for the correct first answer token, plus 0.2 when the completion
    contains only the expected compact answer format. The small format bonus
    discourages verbose clinical explanations during RLVR.
    """

    rewards = []
    for completion, gold, kind in zip(completions, answer, reward_type):
        text = completion_text(completion)
        pred = extract_prediction(text, kind)
        gold_norm = str(gold).strip().upper() if kind == "mcq_letter_exact" else LABEL_MAP.get(str(gold).strip().lower(), str(gold).strip().lower())
        score = 1.0 if pred == gold_norm else 0.0

        compact = text.strip()
        if kind == "mcq_letter_exact" and re.fullmatch(r"[A-Ea-e]\s*[\).:]?", compact):
            score += 0.2
        elif kind == "label_exact" and LABEL_MAP.get(compact.lower().strip(" .:")) == gold_norm:
            score += 0.2
        rewards.append(score)
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/grpo"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gemma2-2b-smlb-grpo"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this script. On Apple Silicon, use this as a template and adapt for MPS/SFT first.")

    data_files = {
        "train": str(args.dataset_dir / "train.jsonl"),
        "validation": str(args.dataset_dir / "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    )

    peft_config = None
    if args.lora_r > 0:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=medical_answer_reward,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
