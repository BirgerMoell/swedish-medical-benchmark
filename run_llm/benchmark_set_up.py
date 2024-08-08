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
    max_tokens: int = 200

    @abstractmethod
    def get_ground_truth(self):
        pass

    @abstractmethod
    def detect_answers(self):
        pass

    @abstractmethod
    def final_prompt_format(self):
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
        self.max_tokens = 10

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

    def final_prompt_format(self, v):
        return self.prompt.format(question=v["QUESTION"])


class GeneralPractioner(Benchmark):
    def __init__(self, prompt: str = ""):
        self.data = json.loads(
            Path(
                "./benchmarks/general_practitioner/general_practioner.json"
            ).read_text()
        )
        self.prompt = prompt
        self.name = "GeneralPractioner"
        self.label_tag_groups = []
        self.answer_options = "options"

    def get_ground_truth(self):
        return np.asarray([v["correct_answer"] for v in self.data.values()])

    def detect_answers(self, llm_answers):
        return np.asarray(llm_answers)
    
    def final_prompt_format(self, v):
        return self.prompt.format(question=v["QUESTION"], options=", ".join(v[self.answer_options]))
    

class SpecialistQuestions(Benchmark):
    def __init__(self, prompt: str = ""):
        self.data = self.__load_data()
        self.prompt = prompt
        self.name = "SpecialistQuestions"
        self.label_tag_groups = ["LABELS"]
        self.answer_options = "options"

    def get_ground_truth(self):
        return np.asarray([v["correct_answer"] for v in self.data.values()])

    def detect_answers(self, llm_answers):
        return np.asarray(llm_answers)
    
    def final_prompt_format(self, v):
        return self.prompt.format(question=v["QUESTION"], options=", ".join(v[self.answer_options]))

    def __load_data(self):
        combined_data = {}
        pathlist = Path("./benchmarks/specalist_questions").glob('*.json')
        for path in pathlist:
            with open(str(path), "r") as file:
                data = json.load(file)
                for i in data.keys():
                    data[i].update({"LABELS": [path.split("/")[-1].split(".")[0]]})
                combined_data.update(data)
        return combined_data


class SwedishDoctorsExam(Benchmark):
    def __init__(self, prompt: str = ""):
        self.data = json.loads(
            Path(
                "./benchmarks/swetheoreticaldoctorsexam/clinical_case.json"
            ).read_text()
        )
        self.prompt = prompt
        self.name = "SwedishDoctorsExam"
        self.label_tag_groups = []

    def get_ground_truth(self):
        return np.asarray([v["ANSWER"].lower() for v in self.data.values()])

    def detect_answers(self, llm_answers):
        return np.asarray([i.strip().lower() for i in llm_answers])
    
    def final_prompt_format(self, v):
        return self.prompt.format(question=v["QUESTION"])


def get_benchmark_by_name(name: str):
    if name == "PubMedQA-L-SWE":
        return PubMedQALSWE()
    elif name == "GeneralPractioner":
        return GeneralPractioner()
    elif name == "SpecialistQuestions":
        return SpecialistQuestions()
    elif name == "SwedishDoctorsExam":
        return SwedishDoctorsExam()
    else:
        raise ValueError(f"Unknown benchmark: {name}")
