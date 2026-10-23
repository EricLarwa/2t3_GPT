"""Prepare a bounded FineWeb-Edu sample with separate cached text splits.

Run: python -m data.fine_web_training
Limits count documents, not tokens. Reuse the checkpoint tokenizer afterward.
Importing this module does not download data.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import datasets


TRAINING_DIR = Path(__file__).resolve().parent / "training" / "fineweb-edu"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SPLITS = ("train", "validation", "test")


def prepare_fineweb_edu(
    split: str = "train", *, train_documents: int = 100_000,
    validation_documents: int = 1_000, test_documents: int = 1_000,
    directory: Path | None = None,
) -> Path:
    """Prepare all splits once and return the requested corpus path.

    A normalized-text hash assigns documents to train/validation/test with
    probabilities 98%/1%/1%. Identical normalized documents cannot cross splits;
    near duplicates and repeated passages are not removed. Each split stops at
    its document limit. Scanning is capped at 20 times the combined limits.

    Settings are recorded in a manifest. Use a new directory for different
    limits to avoid invalidating existing corpora and token caches.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown FineWeb-Edu split: {split}")
    limits = dict(zip(SPLITS, (train_documents, validation_documents, test_documents)))
    if any(type(limit) is not int or limit < 1 for limit in limits.values()):
        raise ValueError("Document limits must be positive integers")
    directory = Path(directory) if directory is not None else TRAINING_DIR
    paths = {name: directory / f"{name}.txt" for name in SPLITS}
    manifest_path = directory / "preparation.json"
    settings = {
        "format_version": 1, "dataset": DATASET_NAME, "config": DATASET_CONFIG,
        "document_limits": limits, "partition": "sha256-normalized-text-98-1-1",
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("settings") != settings:
            raise ValueError("Cached settings differ; choose a new directory")
        if all(path.is_file() and path.stat().st_size for path in paths.values()):
            return paths[split]
        raise ValueError("FineWeb-Edu cache is incomplete; choose a new directory")
    if any(path.exists() or path.with_suffix(".tokens.bin").exists() for path in paths.values()):
        raise ValueError("Untracked corpus files exist; choose a new directory")

    directory.mkdir(parents=True, exist_ok=True)
    documents = datasets.load_dataset(
        DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True,
        cache_dir=str(directory / "cache"),
    )
    temporary = {name: path.with_suffix(".txt.tmp") for name, path in paths.items()}
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    counts = dict.fromkeys(SPLITS, 0)
    print(f"Preparing FineWeb-Edu document limits: {limits}")
    try:
        with ExitStack() as stack:
            outputs = {
                name: stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
                for name, path in temporary.items()
            }
            for scanned, document in enumerate(documents.take(20 * sum(limits.values())), start=1):
                raw_text = document["text"]
                if not isinstance(raw_text, str):
                    raise ValueError("FineWeb-Edu document has a non-string text field")
                text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if not text:
                    continue
                bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % 100
                name = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
                if counts[name] < limits[name]:
                    outputs[name].write(text + "\n\n")
                    counts[name] += 1
                if scanned % 10_000 == 0:
                    print(f"Scanned {scanned:,} documents | Saved: {counts}")
                if counts == limits:
                    break
        if counts != limits:
            raise RuntimeError(f"Stream ended or scan limit reached before filling splits: {counts}")
        for name, path in paths.items():
            temporary[name].replace(path)
        manifest_temporary.write_text(
            json.dumps({"settings": settings, "documents": counts}, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_temporary.replace(manifest_path)
    finally:
        for path in (*temporary.values(), manifest_temporary):
            path.unlink(missing_ok=True)
    print(f"FineWeb-Edu preparation complete: {directory}")
    return paths[split]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-documents", type=int, default=100_000)
    parser.add_argument("--validation-documents", type=int, default=1_000)
    parser.add_argument("--test-documents", type=int, default=1_000)
    parser.add_argument("--directory", type=Path, default=TRAINING_DIR)
    args = parser.parse_args()
    prepare_fineweb_edu(
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
        test_documents=args.test_documents, directory=args.directory,
    )
