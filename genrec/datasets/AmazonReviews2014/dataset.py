"""Adapter for the legacy Amazon 5-core data used by the other baselines.

The official LLaDA-Rec repository only ships an Amazon Reviews 2023 loader.
DiffGRM, MaskGR, and RPG use the older 5-core collections, so this adapter
preserves their chronological sequences and leave-one-out split.  Metadata is
optional when a precomputed semantic-ID file is supplied; placeholder strings
are then sufficient because the recommender never embeds metadata itself.
"""

import ast
import gzip
import json
import os
from collections import defaultdict

from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class AmazonReviews2014(AbstractDataset):

    def __init__(self, config: dict):
        super().__init__(config)
        self.category = config["category"]
        supported = {
            "Beauty",
            "Sports_and_Outdoors",
            "Toys_and_Games",
            "CDs_and_Vinyl",
            "Video_Games",
            "Musical_Instruments",
        }
        if self.category not in supported:
            raise ValueError(
                f"Unsupported Amazon Reviews 2014 category: {self.category}")
        self.cache_dir = os.path.join(
            config["cache_dir"], "AmazonReviews2014", self.category
        )
        self._load_or_process()

    @staticmethod
    def _iter_json_gz(path):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield ast.literal_eval(line)

    @staticmethod
    def _meta_value(value):
        if value is None:
            return ""
        if isinstance(value, list):
            flat = []
            for part in value:
                if isinstance(part, list):
                    flat.extend(str(x) for x in part)
                else:
                    flat.append(str(part))
            value = ", ".join(flat)
        return clean_text(str(value)).strip()

    def _load_or_process(self):
        processed = os.path.join(self.cache_dir, "processed")
        seq_path = os.path.join(processed, "all_item_seqs.json")
        map_path = os.path.join(processed, "id_mapping.json")
        meta_path = os.path.join(processed, "metadata.sentence.json")
        os.makedirs(processed, exist_ok=True)

        if all(os.path.isfile(path) for path in (seq_path, map_path)):
            with open(seq_path, encoding="utf-8") as handle:
                self.all_item_seqs = json.load(handle)
            with open(map_path, encoding="utf-8") as handle:
                self.id_mapping = json.load(handle)
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as handle:
                    self.item2meta = json.load(handle)
            else:
                self.item2meta = {
                    item: f"Item: {item}."
                    for item in self.id_mapping["id2item"][1:]
                }
            return

        reviews_path = os.path.abspath(self.config["local_reviews_path"])
        if not os.path.isfile(reviews_path):
            raise FileNotFoundError(reviews_path)
        metadata_config = self.config.get("local_meta_path")
        metadata_path = (
            os.path.abspath(metadata_config) if metadata_config else None)

        interactions = defaultdict(list)
        for row in tqdm(self._iter_json_gz(reviews_path), desc="Legacy reviews"):
            interactions[row["reviewerID"]].append(
                (row["asin"], int(row["unixReviewTime"]))
            )

        for user, item_times in interactions.items():
            # Stable tie handling reproduces the source-file order.
            item_times.sort(key=lambda pair: pair[1])
            items = [item for item, _ in item_times]
            self.id_mapping["user2id"][user] = len(self.id_mapping["id2user"])
            self.id_mapping["id2user"].append(user)
            for item in items:
                if item not in self.id_mapping["item2id"]:
                    self.id_mapping["item2id"][item] = len(
                        self.id_mapping["id2item"]
                    )
                    self.id_mapping["id2item"].append(item)
            self.all_item_seqs[user] = items

        wanted = set(self.id_mapping["item2id"])
        wanted.discard("[PAD]")
        self.item2meta = {}
        feature_labels = (
            ("title", "Title"),
            ("price", "Price"),
            ("brand", "Brand"),
            ("categories", "Category"),
            ("feature", "Features"),
            ("description", "Description"),
        )
        if metadata_path and os.path.isfile(metadata_path):
            for row in tqdm(
                    self._iter_json_gz(metadata_path),
                    desc="Legacy metadata"):
                item = row.get("asin")
                if item not in wanted:
                    continue
                parts = []
                for key, label in feature_labels:
                    value = self._meta_value(row.get(key))
                    if value:
                        parts.append(f"{label}: {value.rstrip('.')}.")
                self.item2meta[item] = " ".join(parts)

        # SentenceTransformer requires text for every catalog item.
        for item in wanted:
            self.item2meta.setdefault(item, f"Item: {item}.")

        with open(seq_path, "w", encoding="utf-8") as handle:
            json.dump(self.all_item_seqs, handle)
        with open(map_path, "w", encoding="utf-8") as handle:
            json.dump(self.id_mapping, handle)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(self.item2meta, handle)
