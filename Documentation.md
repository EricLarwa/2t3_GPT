
# Oct 16, 2026 - Initial Setup
 Beginning the inital requirments of the application, as usual AI Tech stack initialized as pyTorch, Tenserflow, Numpy, and Hugging Face libraries (Currently only "Tokenizers"). 

### Why This Stack?
-  pyTorch is used to handle the creation of nueral layers, forward passes, backpropogation, and autograd. In simpler terms, pyTorch will handle a significant amount of work
-  Hugging Face's tokenizer library was developed with rust which means, memory efficiency! Woohoooo. This avoids the system using a slow tokenization loop written in python that will assign memory as it feels like. With AI, memory allocation and utilization is of the highest priority. (This project will use Hugging Face's Byte-Pair Encoding *"BPE"*)
-  The project is hyperspecific to its training infrastructure, in this case i will be training on my device *(Intel 9 Ultra 275HX, NVIDIA GeForce RTX 5070 Laptop GPU (8GB), 64 GB of Laptop memory)*. Due to the limitation of my GPU's VRAM, the weight of the model will likely rest at *__~70M - 100M__* parameters, but that's fine i'm not trying to make the next ChatGPT. That being said, i'm shooting for maximum control over memory allocation but the training itself will still be slow as the VRAM holds the weights. 
- TenserFlow and Numpy will be discussed later on, it has not been implemented at this point. 

### What Did We Learn? (WDWL)
- torch uses GPU specific functions to enable specific FMA, i'm using ```torch.set_float32_matmul_precision('high')``` for the 5070's BP16 allowing for faster math kernals
- dropout values define the probability of randomly deactivating a nueron during each training step. For example, if dropout = 0.5 then each nueron has  50% chance of being temporarily set to 0 and excluded from forwarrd and back propogation. 
- Tensors are split into Key, Value, and Query, which makes sense, tensors are scalable to inputs aand it doesn't necessarily have to be one input the entire time. 
- ```.view()``` is pyTorch's grouping of values into dimensions. the line: 
 ```q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2))``` 
 where q becomes (B, T, number_of_heads, size_of_each_head). 