
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
from model.train import get_training_device, train_local_model
from model.checkpoint import load_checkpoint, checkpoint_config
from model.evaluate import evaluate_model, generate_text


def main():
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 and train LocalGPT sequentially.")
    parser.add_argument("--epochs", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue the latest saved training state")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate latest.pt without training")
    parser.add_argument("--best", action="store_true", help="Evaluate best.pt instead of latest.pt")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint interval in batches")
    parser.add_argument("--prompt", help="Text prefix for sampled generation")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.save_every < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error("--save-every, --temperature, and --top-k must be positive")
    if args.best and not args.evaluate_only:
        parser.error("--best requires --evaluate-only")

    checkpoint_dir = TRAINING_DIR / "checkpoints"
    checkpoint_path = checkpoint_dir / ("best.pt" if args.best else "latest.pt")
    checkpoint = None
    if args.resume or args.evaluate_only:
        if not checkpoint_path.is_file():
            parser.error("No checkpoint exists yet; run training first")
        checkpoint = load_checkpoint(checkpoint_path)
    elif checkpoint_path.exists():
        parser.error("A checkpoint already exists. Use --resume to continue or --evaluate-only to test it")
    config = checkpoint_config(checkpoint) if checkpoint is not None else ModelConfig()
    corpus_path = prepare_wikitext2()
    tokenizer_path = TRAINING_DIR / "tokenizer.json"
    if checkpoint is not None:
        tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    elif not tokenizer_path.exists():
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
    validation_path = prepare_wikitext2("validation")
    validation_ids = tokenizer.encode(validation_path.read_text(encoding="utf-8")).ids
    validation = TokenizedDataset(validation_ids, config.block_size, stride=config.block_size)
    model = LocalGPT(config)

    if args.evaluate_only:
        if checkpoint is None:
            raise RuntimeError("A checkpoint is required for evaluation")
        model.load_state_dict(checkpoint["model"])
        model.to(get_training_device())
        metrics = evaluate_model(model, validation, config.batch_size)
        print(f"Validation loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")
    else:
        train_local_model(
            model, dataset, config, epochs=args.epochs, checkpoint_dir=checkpoint_dir,
            resume=checkpoint, tokenizer_json=tokenizer.to_str(), validation_dataset=validation,
            save_every=args.save_every,
        )
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
    prompt_text = args.prompt if args.prompt is not None else tokenizer.decode(prompt[0].tolist())
    print("Prompt:", prompt_text)
    print("Continuation:", generate_text(
        model, tokenizer, prompt_text, temperature=args.temperature, top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
