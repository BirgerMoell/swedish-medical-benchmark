import json
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional


class Benchmark(ABC):
    data: dict
    prompt: str
    labels: Optional[str] = None
    meshes: Optional[str] = None

    @abstractmethod
    def get_ground_truth(self):
        raise NotImplementedError

    @abstractmethod
    def detect_answers(self):
        raise NotImplementedError


class PubMedQALSWE(Benchmark):
    def __init__(self, prompt: str = ""):
        self.data = json.loads(
            Path("./benchmarks/pubmedqa/data/ori_pqal_swe.json").read_text()
        )
        self.prompt = prompt
        self.name = "PubMedQA-L-SWE"
        self.label_tag_groups = ["final_decision", "LABELS", "MESHES"]

    def get_ground_truth(self):
        return np.asarray([v["final_decision"] for v in self.data.values()])

    def detect_answers(self, llm_answers):
        predictions = []
        for answer in llm_answers:
            if "ja" in answer:
                predictions.append("ja")
            elif "nej" in answer:
                predictions.append("nej")
            elif "kanske" in answer:
                predictions.append("kanske")
            else:
                predictions.append("missformat")
        return np.asarray(predictions)


def get_benchmark_by_name(name: str):
    if name == "PubMedQA-L-SWE":
        return PubMedQALSWE()
    else:
        raise ValueError(f"Unknown benchmark {name}")
