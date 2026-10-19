
### Running the connected training pipeline

Install the dependencies in the project environment:

```powershell
.\local_ai_env\Scripts\python.exe -m pip install -r requirements.txt
```

Provide a UTF-8 corpus with more than 512 encoded tokens, then run:

```powershell
.\local_ai_env\Scripts\python.exe main.py --corpus path/to/corpus.txt --epochs 1
```

The default paths are `corpus.txt` and `tokenizer.json` beside `main.py`. Use
`--tokenizer path/to/tokenizer.json` to select a different tokenizer. The first
run trains and saves a tokenizer; later runs load it. The model vocabulary size
comes from that tokenizer. Training uses CUDA with mixed precision when a small
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