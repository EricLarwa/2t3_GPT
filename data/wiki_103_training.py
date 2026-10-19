"""WikiText-103 raw corpus, cached separately from WikiText-2."""

from pathlib import Path

from .wikitext import prepare_split


TRAINING_DIR = Path(__file__).resolve().parent / "training" / "wikitext-103"
BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/"


def split_urls(split: str) -> list[str]:
    if split == "train":
        return [BASE_URL + f"train-{index:05d}-of-00002.parquet" for index in range(2)]
    if split in {"validation", "test"}:
        return [BASE_URL + f"{split}-00000-of-00001.parquet"]
    raise ValueError(f"Unknown WikiText split: {split}")


def prepare_wikitext103(split: str = "train") -> Path:
    return prepare_split(split, TRAINING_DIR, split_urls(split))
