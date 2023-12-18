import gzip
import json
import os
import typing
from collections import defaultdict
import jsonlines
import wget

from src.classes.qaexample import QAExample
from src.utils import BasicTimer, run_ner_linking

ORIG_DATA_DIR = "/data/seongil/datasets/original/"
NORM_DATA_DIR = "/data/seongil/datasets/normalized/"
CUSTOM_DATA_DIR = "/data/seongil/datasets/custom/"

class QADataset(object):
    """
    The base class for Question Answering Datasets that are prepared for 
    substitution functions.
    """

    def __init__(
        self,
        name: str,
        original_path: str,
        preprocessed_path: str,
        custom_path: str = None,
        examples: typing.List[QAExample] = None,
    ):
        """Do not invoke directly. Use `new` or `load`.
        
        Fields:
            name: The name of the dataset --- also used to derive the save path.
            original_path: The original path of the unprocessed data.
            preprocessed_path: The path to the data after processing and saving.
            examples: A list of QAExamples in this dataset. This field is populated by
                `self.read_original_dataset` and later augmented by `self.preprocess`.
        """
        self.name = name
        self.original_path = original_path
        self.preprocessed_path = preprocessed_path
        self.custom_path = custom_path
        self.examples = examples

    @classmethod
    def new(cls, name: str, url_or_path: str):
        """Returns a new QADataset object.

        Args:
            name: Identifying name of this dataset.
            url_or_path: Either the URL to download from, or the local path to read from.
        """
        if os.path.exists(url_or_path):
            original_path = url_or_path
        else:
            file_suffix = ".".join(os.path.basename(url_or_path).split(".")[1:])
            original_path = os.path.join(ORIG_DATA_DIR, f"{name}.{file_suffix}")
            cls._download(name, url_or_path, original_path)
        preprocessed_path = cls._get_norm_dataset_path(name)
        return cls(name=name, original_path=original_path, preprocessed_path=preprocessed_path)

    @classmethod
    def hf_new(cls, name: str, url_or_path: str, split_option="train"):
        from datasets import load_dataset, concatenate_datasets
        original_path = os.path.join(ORIG_DATA_DIR, f"{name}_{split_option}.jsonl.gz")
        if os.path.exists(original_path):
            original_path = original_path
        else:
            dataset = load_dataset(url_or_path)
            if split_option == "all":
                dataset = concatenate_datasets([dataset[split_op] for split_op in dataset.keys()])
            elif split_option =="dev":
                dataset = dataset["validation"]
            elif split_option == "train":
                dataset = dataset["train"]
            elif split_option == "test":
                dataset = dataset["test"]
            else:
                raise NotImplementedError
            dataset.to_json(original_path[:-3])
            with open(original_path[:-3], 'rb') as f_in:
                with gzip.open(original_path, 'wb') as f_out:
                    f_out.writelines(f_in)
        preprocessed_path = os.path.join(NORM_DATA_DIR, f"{name}_{split_option}.jsonl.gz")
        return cls(name=name, original_path=original_path, preprocessed_path=preprocessed_path)

    @classmethod
    def load(cls, name: str):
        """Loads and returns a QADataset object that has already been 
        `self.preprocess` and `self.save()`d

        Args:
            name: Identifying name of this dataset.
        """
        preprocessed_path = cls._get_norm_dataset_path(name)
        assert os.path.exists(
            preprocessed_path
        ), f"Preprocessed dataset should be at {preprocessed_path}."
        with gzip.open(preprocessed_path, "r") as inf:
            header = json.loads(inf.readline())
            if "_all" not in name:
                assert header["dataset"] == name
            examples = [QAExample.json_load(l) for l in inf.readlines()]

        print(f"Read {len(examples)} examples from {preprocessed_path}")
        return cls(name=name, original_path=header["original_path"], preprocessed_path=preprocessed_path, examples=examples)

    @classmethod
    def custom_load(cls, name: str):
        custom_path = os.path.join(CUSTOM_DATA_DIR, f"{name}.jsonl.gz")
        assert os.path.exists(custom_path), f"Custome dataset should be at {custom_path}."
        with gzip.open(custom_path, "r") as inf:
            header = json.loads(inf.readline())
            assert header["dataset"] == name, "Header name doesn't match."
            examples = [QAExample.json_load(l) for l in inf.readlines()]
        print(f"Read {len(examples)} examples from {custom_path}")
        return cls(name, header["original_path"], "", custom_path=custom_path, examples=examples)

    @classmethod
    def _get_norm_dataset_path(self, name: str):
        """Formats the path to the normalized/preprocessed data."""
        return os.path.join(NORM_DATA_DIR, f"{name}.jsonl.gz")

    @classmethod
    def _download(cls, name: str, url: str, dest_path: str):
        """Downloads the original dataset from `url` to `dest_path`."""
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            print(f"Downloading Original Dataset: {name}")
            wget.download(url, dest_path)

    def save(self):
        """Save the preprocessed dataset to JSONL.GZ file. Can be loaded using `self.load()`."""
        os.makedirs(os.path.dirname(self.preprocessed_path), exist_ok=True)
        with gzip.open(self.preprocessed_path, "wt") as outf:
            json.dump({"dataset": self.name, "original_path": self.original_path}, outf)
            outf.write("\n")
            for ex in self.examples:
                json.dump(ex.json_dump(), outf)
                outf.write("\n")
        print(f"Saved preprocessed dataset to {self.preprocessed_path}")

    def custom_save(self):
        self.custom_path = os.path.join(CUSTOM_DATA_DIR, f"{self.name}.jsonl.gz")
        os.makedirs(os.path.dirname(self.custom_path), exist_ok=True)
        with gzip.open(self.custom_path, "wt") as outf:
            json.dump({"dataset": self.name, "original_path": self.original_path}, outf)
            outf.write("\n")
            for ex in self.examples:
                ex.embedding = ex.embedding.tolist()
                json.dump(ex.json_dump(), outf)
                outf.write("\n")
        print(f"Saved custom dataset to {self.custom_path}")

    def read_original_dataset(self, file_path: str):
        """Reads the original/raw dataset into a List of QAExamples.
        
        NB: This is to be implemented by QADataset subclasses, for the specific 
        dataset they represent.
        """
        pass

    def preprocess(
        self, wikidata_info_path: str, ner_model_path: str, debug: bool = False
    ):
        """Read the original dataset, normalize its format and preprocess it. This includes
        running the NER model on the answers, and linking those to wikidata for additional
        metadata that can be used in the logic of answer subsitution functions.

        Args:
            wikidata_info_path: Path to the wikidata entity info saved from Step 1.
            ner_model_path: Path to our SpaCy NER model, downloaded during setup.
            debug: If true, only sample 500 examples to quickly check everything runs end-t-end.
        """
        timer = BasicTimer(f"{self.name} Preprocessing")
        examples = self.read_original_dataset(self.original_path)
        if debug:  # Look at just a subset of examples if debugging
            examples = examples[:500]
        print(f"Processing {len(examples)} Examples...")

        self.label_entities(examples, ner_model_path)
        timer.interval("Labelling and Linking Named Entities")
        self.wikidata_linking(examples, wikidata_info_path)
        timer.interval("Wikidata and Popularity Linking")
        self.examples = examples

        self._report_dataset_stats()
        self.save()
        timer.finish()

    def label_entities(self, examples: typing.List[QAExample], ner_model_path: str):
        """Populate each answer with the NER labels and wikidata ID, if found."""
        all_answers = [answer.text for ex in examples for answer in ex.gold_answers]
        answers_to_info = run_ner_linking(all_answers, ner_model_path)

        for ex in examples:
            for answer in ex.gold_answers:
                # for each match found within the answer
                for ner_info in answers_to_info[answer.text]:
                    if answer.is_equivalent(ner_info["text"]):
                        answer.update_ner_info(
                            ner_info["label"], ner_info["id"]
                        )  # update answer

    def wikidata_linking(
        self, examples: typing.List[QAExample], wikidata_info_path: str
    ):
        """Using the answer's wikidata IDs (if found), extracts wikidata metadata."""
        with gzip.open(wikidata_info_path, "r") as inf:
            wikidata_info = json.load(inf)

        for ex in examples:
            for answer in ex.gold_answers:
                if answer.kb_id in wikidata_info:
                    answer.update_wikidata_info(**wikidata_info[answer.kb_id])

    def _report_dataset_stats(self):
        """Reports basic statistics on what is contained in a preprocessed dataset."""
        grouped_examples = defaultdict(list)
        for ex in self.examples:
            grouped_examples[ex.get_example_answer_type()].append(ex)

        print("Dataset Statistics")
        print("-------------------------------------------")
        print(f"Total Examples = {len(self.examples)}")
        for group, ex_list in grouped_examples.items():
            print(f"Answer Type: {group} | Size of Group: {len(ex_list)}")
        print("-------------------------------------------")

class SquadDataset(QADataset):
    def read_original_dataset(self, file_path: str):
        examples = []
        with gzip.open(file_path, "rb") as file_handle:
                for entry in file_handle:
                    entry = json.loads(entry)
                    examples.append(
                        QAExample.new(
                            uid=entry["id"],
                            query=entry["question"],
                            context=entry["context"],
                            answers=entry["answers"]["text"],
                            title=entry["title"],
                            metadata={},
                        )
                    )
        return examples

class NQ(QADataset):
    def read_original_dataset(self, file_path: str):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                data.append(
                    QAExample.new(
                        uid=idx,
                        query=entry["question"],
                        context=entry["ctxs"][0]["text"],
                        answers=entry["answers"],
                        title=""
                ))
        return data

class Trivia(QADataset):
    def read_original_dataset(self, file_path: str):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                data.append(
                    QAExample.new(
                        uid=idx,
                        query=entry["question"],
                        context=entry["ctxs"][0]["text"],
                        answers=entry["answers"],
                        title=""
                ))
        return data

class MRQANaturalQuetsionsDataset(QADataset):
    """The QADatast for MRQA-Natural Questions.
    
    Original found here: https://github.com/mrqa/MRQA-Shared-Task-2019
    """

    def read_original_dataset(self, file_path: str):
        """Reads the original/raw dataset into a List of QAExamples.
        
        Args:
            file_path: Local path to the dataset.

        Returns:
            List[QAExample]
        """
        examples = []
        with gzip.open(file_path, "rb") as file_handle:
            header = json.loads(file_handle.readline())["header"]
            for entry in file_handle:
                entry = json.loads(entry)
                for qa in entry["qas"]:
                    examples.append(
                        QAExample.new(
                            uid=qa["qid"],
                            query=qa["question"],
                            context=entry["context"],
                            answers=qa["answers"],
                            metadata={},  # NB: Put any metadata you wish saved here.
                        )
                    )
        return examples