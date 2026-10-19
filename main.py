
import argparse
from pathlib import Path

from tokenizers import Tokenizer

from data.dataset import TokenizedDataset
from data.tokenizer import build_and_train_tokenizer
from model.gpt import LocalGPT
from model.model_conf import ModelConfig
from model.train import train_local_model


def main():
    parser = argparse.ArgumentParser(description="Train LocalGPT on a text corpus.")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus.txt")
    parser.add_argument("--tokenizer", type=Path, default=Path(__file__).parent / "tokenizer.json")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if not args.corpus.is_file():
        parser.error(f"Corpus file does not exist: {args.corpus}")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    config = ModelConfig()
    if not args.tokenizer.exists():
        print("No existing tokenizer found. Creating a new one...")
        tokenizer = build_and_train_tokenizer(str(args.corpus), str(args.tokenizer), config.vocab_size)
    else:
        print("Tokenizer already exists. Loading...")
        tokenizer = Tokenizer.from_file(str(args.tokenizer))

    config.vocab_size = tokenizer.get_vocab_size()

    with args.corpus.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Encoding the raw text using the tokenizer...")
    encoded_output = tokenizer.encode(raw_text)
    token_ids = encoded_output.ids
    print(f"Tokenization complete. Number of tokens: {len(token_ids)}")

    dataset = TokenizedDataset(token_ids, config.block_size)
    model = LocalGPT(config)

    train_local_model(model, dataset, config, epochs=args.epochs)


if __name__ == "__main__":
    main()
