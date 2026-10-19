from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from array import array
import hashlib
from itertools import islice
import json
from pathlib import Path

import torch

def build_and_train_tokenizer(input_text_path, save_path, vocab_size=8000):
    """
    Builds and trains a BPE tokenizer on the provided text data.

    Args:
        input_text_path (str): Path to the input text file.
        save_path (str): Path to save the trained tokenizer.
        vocab_size (int): Desired vocabulary size for the tokenizer.

    Reason:
        We use the Rust-backed 'tokenizers' library to leverage all cores of the CPU.
        Writing this in pure Python would take hours; in Rust/C++, it completes a multi-megabyte file in seconds.

    Output:
        Saves a 'tokenizer.json' file to disk which can map raw text strings to integer lists.
    """

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Set the pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the trainer, injecting necessary special control tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    )

    # Trrain the tokenizer using multi-threading
    tokenizer.train(files=[input_text_path], trainer=trainer)

    # save the configuration mapping to disk
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")

    return tokenizer


def encode_corpus(tokenizer: Tokenizer, corpus_path: Path) -> torch.Tensor:
    """Encode Whitespace/BPE text in bounded batches, then map cached int64 IDs."""
    cache = corpus_path.with_suffix(".tokens.bin")
    metadata_path = corpus_path.with_suffix(".tokens.json")
    stat = corpus_path.stat()
    identity = {
        "tokenizer": hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest(),
        "corpus_size": stat.st_size, "corpus_mtime_ns": stat.st_mtime_ns,
        "format": "int64-native-v1",
    }
    metadata = {}
    if cache.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    count = metadata.get("count", -1)
    if metadata.get("identity") != identity or count < 0 or cache.stat().st_size != count * 8:
        temporary = cache.with_suffix(".tmp")
        count = 0
        try:
            with corpus_path.open(encoding="utf-8") as text, temporary.open("wb") as output:
                while lines := list(islice(text, 256)):
                    for encoding in tokenizer.encode_batch(lines):
                        ids = array("q", encoding.ids)
                        ids.tofile(output)
                        count += len(ids)
            temporary.replace(cache)
            temporary_metadata = metadata_path.with_suffix(".tmp")
            temporary_metadata.write_text(json.dumps({"identity": identity, "count": count}), encoding="utf-8")
            temporary_metadata.replace(metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
    return torch.from_file(str(cache), shared=False, size=count, dtype=torch.long)
