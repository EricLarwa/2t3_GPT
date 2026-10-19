from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def build_and_train_tokenizer(input_text_path, save_path, vocab_size=8000):
    """
    Builds and trains a BPE tokenizer on the provided text data.

    Args:
        input_text_path (str): Path to the input text file.
        save_path (str): Path to save the trained tokenizer.
        vocab_size (int): Desired vocabulary size for the tokenizer.

    Reason:
        We use the Rust-backed 'tokenizers' library to leverage all cores of the CPU.
        Writing this in pure Python would take hours; in Rust/C++, it completes a multi-megabyte file in seconds.

    Output:
        Saves a 'tokenizer.json' file to disk which can map raw text strings to integer lists.
    """

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Set the pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the trainer, injecting necessary special control tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    )

    # Trrain the tokenizer using multi-threading
    tokenizer.train(files=[input_text_path], trainer=trainer)

    # save the configuration mapping to disk
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")

    return tokenizer
