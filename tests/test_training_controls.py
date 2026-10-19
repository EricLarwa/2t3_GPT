import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import main
from data import wiki_2_training, wiki_103_training
from data.dataset import TokenizedDataset
from data.tokenizer import build_and_train_tokenizer, encode_corpus
from learning import parsing
from model.checkpoint import load_checkpoint
from model.gpt import LocalGPT
from model.model_conf import ModelConfig
from model.train import train_local_model


class SmallConfig(ModelConfig):
    vocab_size = 64
    block_size = 4
    n_embd = 16
    n_head = 2
    n_layer = 1
    dropout = 0.0
    batch_size = 16


class TrainingControlTests(unittest.TestCase):
    def test_stop_saves_latest_restores_best_and_survives_resume(self):
        config = SmallConfig()
        dataset = TokenizedDataset(list(range(10)), config.block_size)
        model = LocalGPT(config)
        with tempfile.TemporaryDirectory() as directory, \
                patch("torch.cuda.is_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            latest = Path(directory) / "latest.pt"
            with patch("model.train.evaluate_model", side_effect=[
                {"loss": 5.0, "perplexity": 100.0}, {"loss": 5.1, "perplexity": 100.0},
            ]):
                train_local_model(model, dataset, config, epochs=2, checkpoint_dir=directory,
                                  validation_dataset=dataset, early_stopping_patience=2)
            state = load_checkpoint(latest)
            self.assertEqual(state["early_stopping"]["bad_epochs"], 1)
            with patch("model.train.evaluate_model", return_value={"loss": 5.2, "perplexity": 100.0}) as evaluate:
                train_local_model(model, dataset, config, epochs=10, checkpoint_dir=directory,
                                  validation_dataset=dataset, resume=state)
                self.assertEqual(evaluate.call_count, 1)
            stopped = load_checkpoint(latest)
            best = load_checkpoint(Path(directory) / "best.pt")
            self.assertEqual(stopped["epoch"], 3)
            self.assertEqual(stopped["early_stopping"]["bad_epochs"], 2)
            self.assertEqual(best["epoch"], 1)
            for name, weight in model.state_dict().items():
                torch.testing.assert_close(weight, best["model"][name])
            with patch("model.train.evaluate_model") as evaluate:
                train_local_model(model, dataset, config, checkpoint_dir=directory,
                                  validation_dataset=dataset, resume=stopped)
                evaluate.assert_not_called()
            with patch("model.train.evaluate_model", return_value={"loss": 5.3, "perplexity": 100.0}):
                train_local_model(model, dataset, config, epochs=1, checkpoint_dir=directory,
                                  validation_dataset=dataset, resume=stopped, early_stopping_patience=0)
            self.assertEqual(load_checkpoint(latest)["epoch"], 4)

    def test_meaningful_improvement_resets_counter(self):
        config = SmallConfig()
        dataset = TokenizedDataset(list(range(10)), config.block_size)
        losses = [5.0, 4.995, 4.994, 4.98, 4.979, 4.978]
        with tempfile.TemporaryDirectory() as directory, \
                patch("torch.cuda.is_available", return_value=False), \
                patch("model.train.evaluate_model", side_effect=[
                    {"loss": loss, "perplexity": 100.0} for loss in losses
                ]), contextlib.redirect_stdout(io.StringIO()):
            train_local_model(LocalGPT(config), dataset, config, epochs=6, checkpoint_dir=directory,
                              validation_dataset=dataset, early_stopping_patience=3,
                              early_stopping_min_delta=0.01)
            state = load_checkpoint(Path(directory) / "latest.pt")
            self.assertEqual(state["epoch"], 6)
            self.assertEqual(state["early_stopping"]["bad_epochs"], 2)
            self.assertEqual(state["early_stopping"]["best_loss"], 4.98)
            self.assertEqual(state["best_validation_loss"], 4.978)

    def test_chunked_token_cache_matches_full_encoding_and_invalidates(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            corpus = Path(directory) / "train.txt"
            corpus.write_text("hello world\n\nCafé enters the city.\n" * 150, encoding="utf-8")
            tokenizer = build_and_train_tokenizer(str(corpus), str(Path(directory) / "tokenizer.json"), 64)
            expected = tokenizer.encode(corpus.read_text(encoding="utf-8")).ids
            tokens = encode_corpus(tokenizer, corpus)
            self.assertEqual(tokens.tolist(), expected)
            timestamp = corpus.with_suffix(".tokens.bin").stat().st_mtime_ns
            self.assertEqual(encode_corpus(tokenizer, corpus).tolist(), expected)
            self.assertEqual(timestamp, corpus.with_suffix(".tokens.bin").stat().st_mtime_ns)
            corpus.write_text("hello world", encoding="utf-8")
            self.assertEqual(encode_corpus(tokenizer, corpus).tolist(), tokenizer.encode("hello world").ids)

    def test_wikitext103_stage_preserves_source_and_resumes_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            small = Path(directory) / "wikitext-2"
            large = Path(directory) / "wikitext-103"
            argv = ["main.py"]

            def load_rows(*args, **kwargs):
                text = "hello world " * 8
                if "wikitext-103" in str(kwargs["data_files"]):
                    text = "the city hello world " * 8
                return [{"text": text}]

            with patch.object(parsing, "ModelConfig", SmallConfig), \
                    patch.object(parsing, "TRAINING_DIR", small), \
                    patch.object(wiki_2_training, "TRAINING_DIR", small), \
                    patch.object(wiki_103_training, "TRAINING_DIR", large), \
                    patch.object(wiki_2_training.datasets, "load_dataset", side_effect=load_rows) as loader, \
                    patch("sys.argv", argv), patch("torch.cuda.is_available", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                main.main()
                source_path = small / "checkpoints" / "best.pt"
                source_bytes = source_path.read_bytes()
                source = load_checkpoint(source_path)
                argv[:] = ["main.py", "--dataset", "wikitext-103", "--init-from-best"]

                def check_initialization(model, *args, **kwargs):
                    self.assertIsNone(kwargs["resume"])
                    self.assertEqual(kwargs["learning_rate"], 1e-4)
                    for name, weight in model.state_dict().items():
                        torch.testing.assert_close(weight, source["model"][name])
                    return train_local_model(model, *args, **kwargs)

                with patch("learning.data_setup.train_local_model", side_effect=check_initialization):
                    main.main()
                state = load_checkpoint(large / "checkpoints" / "latest.pt")
                self.assertEqual(state["epoch"], 1)
                self.assertEqual(state["tokenizer_json"], source["tokenizer_json"])
                self.assertNotEqual(state["data_info"]["sha256"], source["data_info"]["sha256"])
                train_calls = [c for c in loader.call_args_list if isinstance(c.kwargs["data_files"].get("train"), list)]
                self.assertEqual(train_calls[0].kwargs["data_files"]["train"], wiki_103_training.split_urls("train"))
                self.assertEqual(loader.call_count, 4)
                argv[:] = ["main.py", "--dataset", "wikitext-103", "--resume", "--learning-rate", "0.00005"]
                main.main()
                state = load_checkpoint(large / "checkpoints" / "latest.pt")
                self.assertEqual(state["epoch"], 2)
                self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], 5e-5)
                self.assertEqual(source_path.read_bytes(), source_bytes)
                self.assertEqual(loader.call_count, 4)


if __name__ == "__main__":
    unittest.main()
