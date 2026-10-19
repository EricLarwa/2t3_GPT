
### Running the connected training pipeline

Install the dependencies in the project environment:

```powershell
.\local_ai_env\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare WikiText-2 and train in corpus order with no resource paths:

```powershell
.\local_ai_env\Scripts\python.exe main.py --epochs 1
```

`data/wiki_2_training.py` prepares the maintained `Salesforce/wikitext`
`wikitext-2-raw-v1` training split on the first run. Internet access is needed
for that initial download. Subsequent runs reuse `data/training/wikitext-2/train.txt`.
The downloaded dataset cache and `tokenizer.json` live in the same repository
folder and are excluded from Git. The original archive builder and its comments
are retained in the module for reference; preparation uses the maintained Hub
copy instead of the legacy archive URLs.

The sequence is: prepare training text, train or load the BPE tokenizer, encode
text, create consecutive context chunks, then train LocalGPT. The validation
split is prepared separately for evaluation after each epoch. Validation and
test splits are excluded from tokenizer and model training. Each chunk's targets
are shifted by one token. Training reads chunks in order without shuffling;
any final incomplete chunk is omitted. Each epoch starts at the beginning again.
The model vocabulary size comes from the saved tokenizer.

Training uses CUDA with mixed precision when a small
CUDA operation succeeds and otherwise runs on CPU. If CUDA is detected but its
kernels fail, the trainer reports the error and falls back to CPU. GPU training
requires a PyTorch CUDA build compatible with the installed GPU.
Installing optional `bitsandbytes` enables AdamW8bit
on CUDA; otherwise training uses PyTorch AdamW. Gradient checkpointing is enabled
during training.

Checkpoints are saved under `data/training/wikitext-2/checkpoints/`.
`latest.pt` is replaced every 100 batches and at each epoch end; `best.pt` is
replaced when validation loss improves. Each stores model configuration, weights,
optimizer, mixed precision scaler, epoch/batch position, random state, and the
tokenizer. Writes use a temporary file and atomic replacement. Change the batch
interval with `--save-every 50`.

Continue the latest checkpoint for three additional epochs:

```powershell
.\local_ai_env\Scripts\python.exe main.py --resume --epochs 3
```

A checkpoint from an incomplete epoch first finishes that epoch, counting it as
one of the requested epochs. Resume requires the same training token sequence,
configuration, and optimizer type; its saved tokenizer is authoritative.
To avoid overwriting trained weights, a fresh run stops if a checkpoint already
exists and asks you to use resume or evaluation mode.

Evaluate and sample the best checkpoint without training:

```powershell
.\local_ai_env\Scripts\python.exe main.py --evaluate-only --best --prompt "The history of science"
```

Omit `--best` to evaluate the latest checkpoint. Validation loss and perplexity
cover all complete validation contexts. The training sample score is a separate
small diagnostic, not a generalization score. Generation uses temperature 0.8
and top-k 40 sampling; adjust with `--temperature` and `--top-k`. More varied
sampling does not itself improve model accuracy.

New models initialize embeddings and linear weights with standard deviation
0.02, keeping tied-embedding logits from starting excessively large. Existing
checkpoints retain their learned weights when loaded. Runs from before checkpoint
saving cannot be recovered once their Python process has exited.

SFT is a later stage for task-specific behavior, such as answering prompts. It
requires curated prompt/response examples and an appropriate response-only loss
mask; WikiText pretraining alone is not instruction tuning. Compare validation
loss across pretraining epochs before deciding whether to start that stage.

Run the small integration checks with:

```powershell
.\local_ai_env\Scripts\python.exe -m unittest discover -s tests -v
```
