
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
text, create consecutive context chunks, then train LocalGPT. Validation and
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
during training. This currently trains in memory; model checkpoint saving is
not implemented.

Run the small integration checks with:

```powershell
.\local_ai_env\Scripts\python.exe -m unittest discover -s tests -v
```