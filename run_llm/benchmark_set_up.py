import json
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod


class Benchmark(ABC):
    data: dict
    prompt: str
    labels: str | None = None
    meshes: str | None = None
    og_data: dict | None = None
    answer_options: str | None = None

    @abstractmethod
    def get_ground_truth(self):
        pass

    @abstractmethod
    def detect_answers(self):
        pass


class PubMedQALSWE(Benchmark):
    def __init__(self, prompt: str = ""):
        self.data = json.loads(
            Path("./benchmarks/pubmedqa/data/ori_pqal_swe.json").read_text()
        )
        self.og_data = json.loads(
            Path("./benchmarks/pubmedqa/data/ori_pqal.json").read_text()
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


class GeneralPractioner(Benchmark):
    def __init__(self, prompt: str = ""):
        self.data = json.loads(
            Path(
                "./benchmarks/general_practitioner/general_practioner.json"
            ).read_text()
        )
        self.prompt = prompt
        self.name = "GeneralPractioner"
        self.label_tag_groups = ["correct_answer", "options"]
        self.answer_options = "options"

    def get_ground_truth(self):
        return np.asarray([v["correct_answer"] for v in self.data.values()])

    def detect_answers(self, llm_answers):
        return np.asarray(llm_answers)


def get_benchmark_by_name(name: str):
    if name == "PubMedQA-L-SWE":
        return PubMedQALSWE()
    elif name == "GeneralPractioner":
        return GeneralPractioner()
    else:
        raise ValueError(f"Unknown benchmark: {name}")
