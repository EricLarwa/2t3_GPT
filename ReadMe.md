
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

`learning/parsing.py` returns a typed training setup consumed by
`learning/data_setup.py`; `main.py` calls its `setup_data()` entry point.
After each validation pass, ReduceLROnPlateau checks the current validation loss.
With patience 2, it halves the learning rate after three consecutive
non-improving checks, down to a minimum of `1e-5`. Scheduler history is saved
in checkpoints and restored on resume. Older checkpoints initialize the
scheduler from their saved best validation loss because their earlier plateau
history is unavailable.

Early stopping tracks consecutive validation checks without a meaningful loss
decrease. Defaults are `--patience 8 --min-delta 0.01`. A decrease of more than
0.01 from the early-stopping reference resets the counter; smaller cumulative
decreases can eventually cross that threshold. The lowest observed validation
loss still determines `best.pt`. Stopping state is saved on every checkpoint,
and the final epoch is saved before stopping. Final text samples use the best
saved validation weights, while `latest.pt` retains the actual resume position.
Older checkpoints start with zero unsuccessful checks because that history
was not recorded. A checkpoint already at its stopping limit will not train
again unless the stopping setting is explicitly changed, for example with
`--patience 0` to disable early stopping.

The default dropout is 0.2 for new models. Resuming restores the checkpoint's
dropout setting, so changing `model_conf.py` does not alter existing runs.

Continue the latest checkpoint for three additional epochs:

```powershell
.\local_ai_env\Scripts\python.exe main.py --resume --epochs 3
```

A checkpoint from an incomplete epoch first finishes that epoch, counting it as
one of the requested epochs. Resume requires the same training token sequence,
configuration, and optimizer type; its saved tokenizer is authoritative.
The checkpoint selects its original optimizer even when optional dependencies
have changed: an AdamW checkpoint continues with AdamW on either CPU or CUDA,
even if bitsandbytes was installed later. AdamW8bit checkpoints require CUDA
and bitsandbytes to resume. No optimizer state is discarded automatically.
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

You can also continue the best checkpoint with an explicit lower learning rate:

```powershell
.\local_ai_env\Scripts\python.exe main.py --resume --best --learning-rate 0.0001 --epochs 8
```

The learning-rate override is applied after loading optimizer state.

### WikiText-103

`data/wiki_103_training.py` supports the raw WikiText-103 dataset, including both
training Parquet shards. It downloads each split separately and caches data,
tokenizer artifacts, and checkpoints under `data/training/wikitext-103/`.
WikiText-2 files and checkpoints remain separate.

Start a new WikiText-103 training stage from your best WikiText-2 model:

```powershell
.\local_ai_env\Scripts\python.exe main.py --dataset wikitext-103 --init-from-best --epochs 3
```

This preserves the source weights, architecture, and tokenizer IDs. It starts
new optimizer, scheduler, epoch, and early-stopping state for the new corpus,
with a default learning rate of `0.0001`. It does not modify WikiText-2
checkpoints. The checkpoint's dropout is also preserved. To train a fresh
WikiText-103 model and tokenizer instead, omit `--init-from-best`.

Continue or evaluate the new stage with:

```powershell
.\local_ai_env\Scripts\python.exe main.py --dataset wikitext-103 --resume --epochs 3
.\local_ai_env\Scripts\python.exe main.py --dataset wikitext-103 --evaluate-only --best
```

Both datasets are tokenized in bounded batches into a memory-mapped int64 cache,
avoiding one huge encoding and Python token list for WikiText-103. The cache is
rebuilt when the tokenizer or corpus size/modification time changes. Validation
is encoded with the training tokenizer and never used to train the tokenizer
or update model weights. Corpus order and consecutive context chunks are
preserved. Preparing a dataset does not automatically run model training.

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
