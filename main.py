
from .data.dataset import TokenizedDataset
from .data.tokenizer import build_and_train_tokenizer
from .model.gpt import LocalGPT
from .model.model_config import ModelConfig


if __name__ == "__main__":
    config = ModelConfig()

    if not os.path.exists("tokenizer.json"):
        print("No existing tokenizer found. Creating a new one...")
        tokenizer = build_and_train_tokenizer("corpus.txt", "tokenizer.json", config.vocab_size)
    else:
        print("Tokenizer already exists. Loading...")
        tokenizer = Tokenizer.from_file("tokenizer.json")

    with open("corpus.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Encoding the raw text using the tokenizer...")
    encoded_output = tokenizer.encode(raw_text)
    token_ids = encoded_output.ids
    print(f"Tokenization complete. Number of tokens: {len(token_ids)}")

    dataset = TokenizedDataset(token_ids, config.block_size)
    model = LocalGPT(config)

    train_local_model(model, dataset, config, epochs=1)