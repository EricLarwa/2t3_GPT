"""Prepare separate WikiText splits from the maintained Parquet files."""

from pathlib import Path

import datasets


def prepare_split(split: str, directory: Path, urls: list[str]) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown WikiText split: {split}")
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / f"{split}.txt"
    if corpus_path.is_file() and corpus_path.stat().st_size:
        return corpus_path
    training = datasets.load_dataset(
        "parquet", data_files={split: urls[0] if len(urls) == 1 else urls}, split=split,
        cache_dir=str(directory / "cache"),
        download_config=datasets.DownloadConfig(cache_dir=str(directory / "downloads")),
    )
    temporary_path = corpus_path.with_suffix(".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as corpus:
            for row in training:
                # Default Dataset rows are dicts; inferred formatter types also include arrays.
                corpus.write(row["text"].rstrip("\r\n") + "\n")  # pyright: ignore[reportCallIssue, reportArgumentType]
        temporary_path.replace(corpus_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return corpus_path
