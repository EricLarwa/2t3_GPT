"""Evaluate next-token predictions and sample text without changing model weights."""

import math

import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate_model(model, dataset, batch_size):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    try:
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            count = y.numel()
            total_loss += loss.item() * count
            total_tokens += count
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Evaluation dataset contains no complete contexts")
    loss = total_loss / total_tokens
    perplexity = math.exp(loss) if loss < 709 else math.inf
    return {"loss": loss, "perplexity": perplexity, "tokens": total_tokens}


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    if temperature <= 0 or top_k < 1 or max_new_tokens < 0:
        raise ValueError("temperature and top_k must be positive; max_new_tokens cannot be negative")
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("Prompt must encode to at least one token")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(generated[:, -model.config.block_size:])
            logits = logits[:, -1, :].float() / temperature
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            sampled = torch.multinomial(torch.softmax(values, dim=-1), num_samples=1)
            next_token = indices.gather(-1, sampled)
            generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)
    return tokenizer.decode(generated[0].tolist())
