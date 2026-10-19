import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer

import main
from data.dataset import TokenizedDataset
from model.gpt import LocalGPT
from model.model_conf import ModelConfig
from model.train import get_training_device, train_local_model


class TinyConfig(ModelConfig):
    vocab_size = 64
    block_size = 4
    n_embd = 16
    n_head = 2
    n_layer = 1
    dropout = 0.0
    batch_size = 16


class IntegrationTests(unittest.TestCase):
    def test_unusable_cuda_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.ones", side_effect=RuntimeError("no kernel image")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(get_training_device(), torch.device("cpu"))

    def test_entrypoint_trains_with_new_and_saved_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            tokenizer_path = Path(directory) / "tokenizer.json"
            corpus.write_text("hello world hello world hello world hello world", encoding="utf-8")
            argv = ["main.py", "--corpus", str(corpus), "--tokenizer", str(tokenizer_path)]
            with patch.object(main, "ModelConfig", TinyConfig), patch("sys.argv", argv), \
                    patch("torch.cuda.is_available", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                main.main()
                main.main()
            self.assertEqual(output.getvalue().count("Starting training..."), 2)
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.assertGreater(tokenizer.get_vocab_size(), 7)

    def test_causal_attention_and_inference(self):
        torch.manual_seed(0)
        model = LocalGPT(TinyConfig()).eval()
        x = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 5, 6]])
        with torch.no_grad():
            logits, loss = model(x, x)
            changed_logits, _ = model(changed, changed)
            final_logits, no_loss = model(x)
        torch.testing.assert_close(logits[:, :2], changed_logits[:, :2])
        torch.testing.assert_close(final_logits, logits[:, -1:])
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(no_loss)
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)

    def test_training_updates_weights_and_reaches_all_layers(self):
        torch.manual_seed(0)
        model = LocalGPT(TinyConfig())
        dataset = TokenizedDataset(list(range(10)), TinyConfig.block_size)
        before = model.lm_head.weight.detach().clone()
        with patch("torch.cuda.is_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            train_local_model(model, dataset, TinyConfig(), epochs=1)
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_dataset_targets_and_short_corpus(self):
        dataset = TokenizedDataset([0, 1, 2, 3, 4], 4)
        x, y = dataset[0]
        self.assertEqual(x.tolist(), [0, 1, 2, 3])
        self.assertEqual(y.tolist(), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            TokenizedDataset([0, 1], 4)


if __name__ == "__main__":
    unittest.main()
