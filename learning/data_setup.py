
from dataclasses import dataclass
import math
import torch
from torch.utils.data import DataLoader

from data.dataset import TokenizedDataset
from data.wiki_2_training import prepare_wikitext2
from data.wiki_103_training import prepare_wikitext103
from data.tokenizer import encode_corpus
from model.evaluate import evaluate_model, generate_text
from model.gpt import LocalGPT
from model.train import get_training_device, train_local_model
from .parsing import TrainingSetup, parsers

@dataclass
class DataSetup:
    """
    A dataclass to hold the setup for training or evaluating the model.
    """
    setup: TrainingSetup
    dataset: TokenizedDataset
    validation: TokenizedDataset
    model: LocalGPT

def setup_data() -> DataSetup:
    setup = parsers()

    dataset = TokenizedDataset(setup.token_ids, setup.config.block_size, stride=setup.config.block_size)
    prepare_corpus = prepare_wikitext2 if setup.args.dataset == "wikitext-2" else prepare_wikitext103
    validation_path = prepare_corpus("validation")
    validation_ids = encode_corpus(setup.tokenizer, validation_path)
    validation = TokenizedDataset(validation_ids, setup.config.block_size, stride=setup.config.block_size)
    model = LocalGPT(setup.config)
    if setup.initialization is not None:
        model.load_state_dict(setup.initialization["model"])

    if setup.args.evaluate_only:
        if setup.checkpoint is None:
            raise RuntimeError("A checkpoint is required for evaluation")
        model.load_state_dict(setup.checkpoint["model"])
        model.to(get_training_device())
        metrics = evaluate_model(model, validation, setup.config.batch_size)
        print(f"Validation loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")
    else:
        train_local_model(
            model, dataset, setup.config, epochs=setup.args.epochs, checkpoint_dir=setup.checkpoint_dir,
            resume=setup.checkpoint, tokenizer_json=setup.tokenizer.to_str(), validation_dataset=validation,
            save_every=setup.args.save_every,
            early_stopping_patience=setup.args.patience,
            early_stopping_min_delta=setup.args.min_delta,
            learning_rate=setup.args.learning_rate,
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
    prompt = x[:2, :7]
    prompt_text = setup.args.prompt if setup.args.prompt is not None else setup.tokenizer.decode(prompt[0].tolist())
    print("Prompt:", prompt_text)
    print("Continuation:", generate_text(
        model, setup.tokenizer, prompt_text, temperature=setup.args.temperature, top_k=setup.args.top_k,
    ))
    return DataSetup(
        setup=setup,
        dataset=dataset,
        validation=validation,
        model=model,
    )
