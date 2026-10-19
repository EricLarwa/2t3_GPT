"""
Parsing utilities for setting up the training environment.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
import torch

from data.tokenizer import build_and_train_tokenizer, encode_corpus
from data.wiki_2_training import TRAINING_DIR, prepare_wikitext2
from data import wiki_103_training
from model.checkpoint import checkpoint_config, load_checkpoint
from model.model_conf import ModelConfig

@dataclass
class TrainingSetup:
    args: argparse.Namespace
    config: ModelConfig
    checkpoint: dict[str, Any] | None
    checkpoint_dir: Path
    tokenizer: Tokenizer
    token_ids: torch.Tensor
    initialization: dict[str, Any] | None = None


def parsers() -> TrainingSetup:
    parser = argparse.ArgumentParser(description="Prepare WikiText and train LocalGPT sequentially.")
    parser.add_argument("--dataset", choices=("wikitext-2", "wikitext-103"), default="wikitext-2")
    parser.add_argument("--epochs", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue the latest saved training state")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate latest.pt without training")
    mode.add_argument("--init-from-best", action="store_true", help="Start WikiText-103 from the best WikiText-2 weights")
    parser.add_argument("--best", action="store_true", help="Use best.pt for resume or evaluation")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint interval in batches")
    parser.add_argument("--prompt", help="Text prefix for sampled generation")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--patience", type=int, help="Checks without improvement before stopping; 0 disables (default: 8)")
    parser.add_argument("--min-delta", type=float, help="Meaningful validation loss decrease (default: 0.01)")
    parser.add_argument("--learning-rate", type=float, help="Override LR after restoring optimizer state")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.save_every < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error("--save-every, --temperature, and --top-k must be positive")
    if args.best and not (args.evaluate_only or args.resume):
        parser.error("--best requires --evaluate-only or --resume")
    if args.init_from_best and args.dataset != "wikitext-103":
        parser.error("--init-from-best requires --dataset wikitext-103")
    if args.patience is not None and args.patience < 0:
        parser.error("--patience must be non-negative")
    if args.min_delta is not None and not 0 <= args.min_delta < float("inf"):
        parser.error("--min-delta must be finite and non-negative")
    if args.learning_rate is not None and not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive")

    training_dir = TRAINING_DIR if args.dataset == "wikitext-2" else wiki_103_training.TRAINING_DIR
    prepare_corpus = prepare_wikitext2 if args.dataset == "wikitext-2" else wiki_103_training.prepare_wikitext103
    checkpoint_dir = training_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / ("best.pt" if args.best else "latest.pt")
    checkpoint = None
    if args.resume or args.evaluate_only:
        if not checkpoint_path.is_file():
            parser.error("No checkpoint exists yet; run training first")
        checkpoint = load_checkpoint(checkpoint_path)
    elif checkpoint_path.exists():
        parser.error("A checkpoint already exists. Use --resume to continue or --evaluate-only to test it")
    initialization = None
    if args.init_from_best:
        source_path = TRAINING_DIR / "checkpoints" / "best.pt"
        if not source_path.is_file():
            parser.error("No WikiText-2 best checkpoint exists to initialize WikiText-103")
        initialization = load_checkpoint(source_path)
        if args.learning_rate is None:
            args.learning_rate = 1e-4
    source = checkpoint if checkpoint is not None else initialization
    config = checkpoint_config(source) if source is not None else ModelConfig()
    corpus_path = prepare_corpus()
    tokenizer_path = training_dir / "tokenizer.json"
    if source is not None:
        tokenizer = Tokenizer.from_str(source["tokenizer_json"])
        if initialization is not None:
            tokenizer.save(str(tokenizer_path))
    elif not tokenizer_path.exists():
        print("No existing tokenizer found. Creating a new one...")
        tokenizer = build_and_train_tokenizer(str(corpus_path), str(tokenizer_path), config.vocab_size)
    else:
        print("Tokenizer already exists. Loading...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    config.vocab_size = tokenizer.get_vocab_size()

    print(f"Encoding the raw text using the tokenizer...")
    token_ids = encode_corpus(tokenizer, corpus_path)
    print(f"Tokenization complete. Number of tokens: {len(token_ids)}")
    return TrainingSetup(args, config, checkpoint, checkpoint_dir, tokenizer, token_ids, initialization)
