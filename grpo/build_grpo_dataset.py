#!/usr/bin/env python3
"""Build Swedish medical GRPO/RLVR data from benchmark sources.

The default build intentionally excludes translated MedQA and PubMedQA. It uses
native Swedish multiple-choice and short-label tasks where rewards can be
computed deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceSpec:
    benchmark: str
    path: str
    loader: str
    domain: str
    trust_tier: str = "native_swedish"
    translated: bool = False
    source_license: str = "See repository source metadata"


@dataclass
class GrpoExample:
    id: str
    benchmark: str
    source_file: str
    language: str
    domain: str
    task_type: str
    trust_tier: str
    prompt: str
    question: str
    answer: str
    answer_text: str | None
    answer_letter: str | None
    options: list[str]
    reward_type: str
    split: str
    source_license: str
    source_notes: str


NATIVE_SOURCES = [
    SourceSpec(
        benchmark="swedish_doctors_exam",
        path="benchmarks/swetheoreticaldoctorsexam/clinical_case.json",
        loader="doctor_exam",
        domain="medical_exam",
    ),
    SourceSpec(
        benchmark="emergency_medicine",
        path="benchmarks/specialist_questions/emergency_medicine/emergency_medicine_clinical_format.json",
        loader="clinical_format",
        domain="emergency_medicine",
    ),
    SourceSpec(
        benchmark="general_medicine",
        path="benchmarks/specialist_questions/gp/fall_description_clinical_format.json",
        loader="clinical_format",
        domain="general_medicine",
    ),
    SourceSpec(
        benchmark="general_practitioner",
        path="benchmarks/general_practitioner/general_practioner.json",
        loader="specialist_list",
        domain="general_practice",
    ),
    SourceSpec(
        benchmark="anesthesiology",
        path="benchmarks/specialist_questions/anestesi.json",
        loader="specialist_list",
        domain="anesthesiology",
    ),
    SourceSpec(
        benchmark="cardiology",
        path="benchmarks/specialist_questions/cardiology.json",
        loader="specialist_list",
        domain="cardiology",
    ),
    SourceSpec(
        benchmark="dermatology",
        path="benchmarks/specialist_questions/dermatology.json",
        loader="specialist_list",
        domain="dermatology",
    ),
    SourceSpec(
        benchmark="endocrinology",
        path="benchmarks/specialist_questions/endocrinology.json",
        loader="specialist_list",
        domain="endocrinology",
    ),
    SourceSpec(
        benchmark="hematology",
        path="benchmarks/specialist_questions/hematologi.json",
        loader="specialist_list",
        domain="hematology",
    ),
    SourceSpec(
        benchmark="neurology",
        path="benchmarks/specialist_questions/neurologi.json",
        loader="specialist_list",
        domain="neurology",
    ),
    SourceSpec(
        benchmark="psychiatry",
        path="benchmarks/specialist_questions/psychiatri.json",
        loader="specialist_list",
        domain="psychiatry",
    ),
]


OPTION_RE = re.compile(r"(?im)^\s*([A-Ea-e])[\).:]\s+(.+?)(?=^\s*[A-Ea-e][\).:]|\Z)", re.S)
ANSWER_LETTER_RE = re.compile(r"\b([A-Ea-e])\s*[\).:]?")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def make_id(benchmark: str, index: int, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    return f"{benchmark}-{index:05d}-{digest}"


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def answer_letter(answer: str) -> str | None:
    match = ANSWER_LETTER_RE.search(answer or "")
    if not match:
        return None
    letter = match.group(1).upper()
    return letter if letter in {"A", "B", "C", "D", "E"} else None


def answer_text(answer: str) -> str | None:
    text = re.sub(r"^\s*[A-Ea-e]\s*[\).:]\s*", "", answer or "").strip()
    return normalize_ws(text) or None


def parse_options_from_question(question: str) -> list[str]:
    options = []
    for letter, body in OPTION_RE.findall(question or ""):
        cleaned = normalize_ws(body)
        if cleaned:
            options.append(f"{letter.upper()}) {cleaned}")
    return options


def normalize_options(options: Any, question: str) -> list[str]:
    if isinstance(options, dict):
        return [f"{str(k).upper()}) {normalize_ws(v)}" for k, v in options.items()]
    if isinstance(options, list):
        cleaned = []
        for index, option in enumerate(options):
            text = normalize_ws(option)
            if not text:
                continue
            if re.match(r"^[A-Ea-e]\s*[\).:]", text):
                cleaned.append(text)
            elif index < 5:
                cleaned.append(f"{chr(ord('A') + index)}) {text}")
        if cleaned:
            return cleaned
    return parse_options_from_question(question)


def infer_answer_letter_from_options(raw_answer: str, options: list[str]) -> str | None:
    target = normalize_ws(answer_text(raw_answer) or raw_answer).lower()
    if not target:
        return None
    for index, option in enumerate(options):
        option_body = normalize_ws(re.sub(r"^\s*[A-Ea-e]\s*[\).:]\s*", "", option)).lower()
        if option_body == target and index < 5:
            return chr(ord("A") + index)
    return None


def mcq_prompt(question: str, options: list[str]) -> str:
    option_block = ""
    if options and not any(option in question for option in options[:1]):
        option_block = "\n\nSvarsalternativ:\n" + "\n".join(options)
    return (
        "Du är en medicinsk frågebesvarare. Välj det bästa svarsalternativet.\n"
        "Svara endast med en bokstav: A, B, C, D eller E. Lägg inte till någon förklaring.\n\n"
        f"Fråga:\n{question}{option_block}\n\nSvar:"
    )


def label_prompt(question: str, labels: Iterable[str]) -> str:
    label_text = ", ".join(labels)
    return (
        "Du är en medicinsk frågebesvarare. Välj den etikett som bäst besvarar frågan.\n"
        f"Svara endast med ett av följande ord: {label_text}. Lägg inte till någon förklaring.\n\n"
        f"Fråga:\n{question}\n\nSvar:"
    )


def make_mcq_example(
    spec: SourceSpec,
    source_path: Path,
    index: int,
    question: str,
    raw_answer: str,
    options: list[str],
) -> GrpoExample | None:
    question = normalize_ws(question)
    raw_answer = normalize_ws(raw_answer)
    letter = answer_letter(raw_answer) or infer_answer_letter_from_options(raw_answer, options)
    if not question or not raw_answer or not letter:
        return None
    ex_id = make_id(spec.benchmark, index, question)
    return GrpoExample(
        id=ex_id,
        benchmark=spec.benchmark,
        source_file=str(source_path.relative_to(ROOT)),
        language="sv",
        domain=spec.domain,
        task_type="medical_mcq",
        trust_tier=spec.trust_tier,
        prompt=mcq_prompt(question, options),
        question=question,
        answer=letter,
        answer_text=answer_text(raw_answer),
        answer_letter=letter,
        options=options,
        reward_type="mcq_letter_exact",
        split="",
        source_license=spec.source_license,
        source_notes="Generated from Swedish Medical Benchmark source file.",
    )


def load_doctor_exam(spec: SourceSpec, source_path: Path) -> list[GrpoExample]:
    rows = read_json(source_path)
    examples = []
    for index, row in enumerate(rows.values() if isinstance(rows, dict) else rows):
        question = row.get("QUESTION")
        answer = row.get("ANSWER")
        options = parse_options_from_question(question or "")
        ex = make_mcq_example(spec, source_path, index, question, answer, options)
        if ex:
            examples.append(ex)
    return examples


def load_clinical_format(spec: SourceSpec, source_path: Path) -> list[GrpoExample]:
    rows = read_json(source_path)
    iterable = rows.values() if isinstance(rows, dict) else rows
    examples = []
    for index, row in enumerate(iterable):
        question = row.get("Question")
        answer = row.get("Answer")
        options = parse_options_from_question(question or "")
        ex = make_mcq_example(spec, source_path, index, question, answer, options)
        if ex:
            examples.append(ex)
    return examples


def load_specialist_list(spec: SourceSpec, source_path: Path) -> list[GrpoExample]:
    rows = read_json(source_path)
    iterable = rows.values() if isinstance(rows, dict) else rows
    examples = []
    for index, row in enumerate(iterable):
        question = row.get("question") or row.get("Question")
        answer = row.get("correct_answer") or row.get("answer") or row.get("Answer")
        options = normalize_options(row.get("options"), question or "")
        ex = make_mcq_example(spec, source_path, index, question, answer, options)
        if ex:
            examples.append(ex)
    return examples


def load_pubmedqa(spec: SourceSpec, source_path: Path) -> list[GrpoExample]:
    rows = read_json(source_path)
    label_map = {"yes": "ja", "no": "nej", "maybe": "kanske", "ja": "ja", "nej": "nej", "kanske": "kanske"}
    examples = []
    iterable = rows.items() if isinstance(rows, dict) else enumerate(rows)
    for raw_index, row in iterable:
        question = normalize_ws(row.get("QUESTION") or row.get("question"))
        contexts = row.get("CONTEXTS") or row.get("contexts") or []
        context = "\n".join(normalize_ws(c) for c in contexts if normalize_ws(c))
        answer = normalize_ws(row.get("final_decision") or row.get("answer") or row.get("LABEL"))
        answer = label_map.get(answer.lower())
        if not question or not answer:
            continue
        full_question = f"{context}\n\nFråga: {question}" if context else question
        ex_id = make_id(spec.benchmark, int(raw_index) if str(raw_index).isdigit() else len(examples), full_question)
        examples.append(
            GrpoExample(
                id=ex_id,
                benchmark=spec.benchmark,
                source_file=str(source_path.relative_to(ROOT)),
                language="sv",
                domain=spec.domain,
                task_type="medical_pubmedqa",
                trust_tier=spec.trust_tier,
                prompt=label_prompt(full_question, ["ja", "nej", "kanske"]),
                question=full_question,
                answer=answer,
                answer_text=answer,
                answer_letter=None,
                options=["ja", "nej", "kanske"],
                reward_type="label_exact",
                split="",
                source_license=spec.source_license,
                source_notes="Translated PubMedQA. Excluded by default; include with --include-translated-pubmedqa.",
            )
        )
    return examples


LOADERS = {
    "doctor_exam": load_doctor_exam,
    "clinical_format": load_clinical_format,
    "specialist_list": load_specialist_list,
    "pubmedqa": load_pubmedqa,
}


def split_name(example_id: str, seed: int, validation_fraction: float, test_fraction: float) -> str:
    value = stable_hash(example_id, seed)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def build_examples(args: argparse.Namespace) -> list[GrpoExample]:
    specs = list(NATIVE_SOURCES)
    if args.include_translated_pubmedqa:
        specs.append(
            SourceSpec(
                benchmark="pubmedqa_swedish_translated",
                path="benchmarks/pubmedqa/data/ori_pqal_swe.json",
                loader="pubmedqa",
                domain="biomedical_literature",
                trust_tier="translated_pubmedqa",
                translated=True,
            )
        )

    examples: list[GrpoExample] = []
    for spec in specs:
        source_path = ROOT / spec.path
        if not source_path.exists():
            print(f"warning: missing source {spec.path}")
            continue
        loaded = LOADERS[spec.loader](spec, source_path)
        examples.extend(loaded)

    random.Random(args.seed).shuffle(examples)
    for ex in examples:
        ex.split = split_name(ex.id, args.seed, args.validation_fraction, args.test_fraction)
    return examples


def write_jsonl(path: Path, rows: Iterable[GrpoExample]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--include-translated-pubmedqa", action="store_true")
    args = parser.parse_args()

    if args.validation_fraction < 0 or args.test_fraction < 0:
        raise SystemExit("split fractions must be non-negative")
    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("validation + test fractions must be below 1")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_examples(args)
    by_split = {name: [ex for ex in examples if ex.split == name] for name in ["train", "validation", "test"]}

    counts = {split: write_jsonl(output_dir / f"{split}.jsonl", rows) for split, rows in by_split.items()}
    manifest = {
        "name": "swedish-medical-grpo",
        "description": "Deterministic GRPO/RLVR dataset for Swedish medical multiple-choice and label tasks.",
        "language": "sv",
        "seed": args.seed,
        "counts": counts,
        "benchmarks": dict(Counter(ex.benchmark for ex in examples)),
        "reward_types": dict(Counter(ex.reward_type for ex in examples)),
        "included_translated_pubmedqa": args.include_translated_pubmedqa,
        "excluded_by_default": ["benchmarks/medqa-swe"],
        "caution": "For medical education/reasoning experiments only; not a clinical deployment dataset.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
