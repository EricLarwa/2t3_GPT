
import argparse
import math
import torch
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from data.dataset import TokenizedDataset
from data.tokenizer import build_and_train_tokenizer
from data.wiki_2_training import TRAINING_DIR, prepare_wikitext2
from model.gpt import LocalGPT
from model.model_conf import ModelConfig
from model.train import train_local_model


def main():
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 and train LocalGPT sequentially.")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    config = ModelConfig()
    corpus_path = prepare_wikitext2()
    tokenizer_path = TRAINING_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        print("No existing tokenizer found. Creating a new one...")
        tokenizer = build_and_train_tokenizer(str(corpus_path), str(tokenizer_path), config.vocab_size)
    else:
        print("Tokenizer already exists. Loading...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    config.vocab_size = tokenizer.get_vocab_size()

    with corpus_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Encoding the raw text using the tokenizer...")
    encoded_output = tokenizer.encode(raw_text)
    token_ids = encoded_output.ids
    print(f"Tokenization complete. Number of tokens: {len(token_ids)}")

    dataset = TokenizedDataset(token_ids, config.block_size, stride=config.block_size)
    model = LocalGPT(config)

    train_local_model(model, dataset, config, epochs=args.epochs)
    model.eval()  # Disable dropout for evaluation
    device = next(model.parameters()).device

    # Score a small batch from the training corpus.
    x, y = next(iter(DataLoader(dataset, batch_size=4)))
    x, y = x.to(device), y.to(device)

    with torch.inference_mode():
        _, loss = model(x, y)

    print(f"Training sample loss: {loss.item():.4f}")
    print(f"Training sample perplexity: {math.exp(loss.item()):.2f}")

    # Give the model a corpus prefix and let it continue.
    prompt = x[:1, :8]
    generated = prompt.clone()

    with torch.inference_mode():
        for _ in range(50):
            context = generated[:, -config.block_size:]
            logits, _ = model(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)

    print("Prompt:", tokenizer.decode(prompt[0].tolist()))
    print("Continuation:", tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
