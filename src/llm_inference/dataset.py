import csv
import json
import math
from typing import Dict, Generator, List, Union

import pandas as pd

from .utils import TemplateLoader


class ClaimPostDataset:
    def __init__(
        self,
        fact_checks_csv: str,
        posts_csv: str,
        mapping_csv: str,
        retrieval_json: str,
        template_name: str,
        batch_size: int = 1,
    ):

        self.batch_size = batch_size
        self.template = TemplateLoader().load(template_name)

        # Load fact checks into a dictionary: fact_check_id -> content
        self.fact_checks = {}
        with open(fact_checks_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.fact_checks[int(row["fact_check_id"])] = row["content"]

        # Load posts into a dictionary: post_id -> content
        self.posts = {}
        with open(posts_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.posts[int(row["post_id"])] = row["content"]

        # Load positive mappings into a set of (fact_check_id, post_id)
        self.positive_pairs = set()
        with open(mapping_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.positive_pairs.add((int(row["fact_check_id"]), int(row["post_id"])))

        # Load top-k retrievals
        with open(retrieval_json, "r", encoding="utf-8") as f:
            self.retrieval_data = json.load(f)

        # Flatten into list of tuples: (fact_check_id, post_id, relevance)
        self.data = []
        for post_id_str, matches in self.retrieval_data.items():
            post_id = int(post_id_str)
            for match in matches:
                fact_check_id = match["id"]
                relevance = (fact_check_id, post_id) in self.positive_pairs
                self.data.append((fact_check_id, post_id, relevance))

    def __iter__(self) -> Generator[List[Dict[str, Union[str, bool]]], None, None]:
        batch = []
        for fact_check_id, post_id, relevance in self.data:
            claim = self.fact_checks.get(fact_check_id, "")
            post = self.posts.get(post_id, "")
            sample = {"claim": claim, "post": post, "relevance": relevance}
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def filter_indices(self, indices: List[int]) -> None:
        self.data = [self.data[i] for i in indices]

    @property
    def num_batches(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def __len__(self) -> int:
        return self.num_batches
