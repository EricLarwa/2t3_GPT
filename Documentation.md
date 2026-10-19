
# Oct 16, 2026 - Initial Setup
 Beginning the inital requirments of the application, as usual AI Tech stack initialized as pyTorch, Tenserflow, Numpy, and Hugging Face libraries (Currently only "Tokenizers").

### Why This Stack?
-  pyTorch is used to handle the creation of nueral layers, forward passes, backpropogation, and autograd. In simpler terms, pyTorch will handle a significant amount of work
-  Hugging Face's tokenizer library was developed with rust which means, memory efficiency! Woohoooo. This avoids the system using a slow tokenization loop written in python that will assign memory as it feels like. With AI, memory allocation and utilization is of the highest priority. (This project will use Hugging Face's Byte-Pair Encoding *"BPE"*)
-  The project is hyperspecific to its training infrastructure, in this case i will be training on my device *(Intel 9 Ultra 275HX, NVIDIA GeForce RTX 5070 Laptop GPU (8GB), 64 GB of Laptop memory)*. Due to the limitation of my GPU's VRAM, the weight of the model will likely rest at *__~70M - 100M__* parameters, but that's fine i'm not trying to make the next ChatGPT. That being said, i'm shooting for maximum control over memory allocation but the training itself will still be slow as the VRAM holds the weights.
- TenserFlow and Numpy will be discussed later on, it has not been implemented at this point.

### What Did We Learn? (WDWL)
- torch uses GPU specific functions to enable specific FMA, i'm using ```torch.set_float32_matmul_precision('high')``` for the 5070's BP16 allowing for faster math kernals
- dropout values define the probability of randomly deactivating a nueron during each training step. For example, if dropout = 0.5 then each nueron has a 50% chance of being temporarily set to 0 and excluded from forward and back propogation.
- Tensors are split into Key, Value, and Query, which makes sense, tensors are scalable to inputs and it doesn't necessarily have to be one input the entire time.
- ```.view()``` is pyTorch's grouping of values into dimensions. the line:
 ```q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2))```
 where q becomes (B, T, number_of_heads, size_of_each_head).

###
# Oct 19, 2026 - Transformer Layer
Implementation of the transformer block that will chain the attention engine together with a "feed-forward" network, using a Pre-layer normalization architecture for stable training.

### Why Do We Need It?
- Transformers are the latest adaptation to the AI cycle (kind of, Google's researchers made this around 2017...I think). Transformers were brought in to solve the sequential processing bottleneck and long-range context limitations of older Nueral Networks.

### WDWL
- HuggingFace's Tokenizer is super easy to use. It's doing the majority (if not all) of the work inside of ```tokenizer.py``` I mean it even splits the whitespace for you.
- Byte-Pair Encoding(BPE) breaks down text into manageable subwords units rather than whole words or individual characters. It also conserves the context window by grouping the subwords, meaning shorter token sequences than character-by-character splitting which leads to longer pieces of text within their memory limits. Boom, memory optimization son!
- GradScaler, singificantly useful when your memory is low. Since gradients can occasionaly become too microscopic and rround to zero, it will dynamically multiply gradient sizes to protect arithmatic accurracy

### WikiText-2 training pipeline

`main.py` now calls `data/wiki_2_training.py` to prepare WikiText-2 automatically.
Run `python main.py --epochs 1` from the project environment; no corpus or
tokenizer path arguments are required. Data and tokenizer artifacts are stored
under `data/training/wikitext-2/` and reused on later runs.

Training uses consecutive context chunks with one-token-shifted targets and
shuffling disabled. Only the training split is downloaded and used. See
[ReadMe.md](ReadMe.md) for setup and runtime details.
