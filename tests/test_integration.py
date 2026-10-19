import contextlib
import io
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, call, Mock

import torch
from tokenizers import Tokenizer

import main
from learning import parsing as parser_module
from data import wiki_2_training
from data.dataset import TokenizedDataset
from model.gpt import LocalGPT
from model.model_conf import ModelConfig
from model.train import create_optimizer, get_training_device, train_local_model
from model.checkpoint import load_checkpoint, save_checkpoint, checkpoint_config
from model.evaluate import evaluate_model, generate_text


class TinyConfig(ModelConfig):
    vocab_size = 64
    block_size = 4
    n_embd = 16
    n_head = 2
    n_layer = 1
    dropout = 0.0
    batch_size = 16


class IntegrationTests(unittest.TestCase):
    def test_saved_adamw_ignores_newly_available_bitsandbytes(self):
        model = LocalGPT(TinyConfig())
        bnb = Mock()
        with patch.dict(sys.modules, {"bitsandbytes": bnb}):
            optimizer = create_optimizer(model, torch.device("cuda"), "AdamW")
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        bnb.optim.AdamW8bit.assert_not_called()

    def test_saved_8bit_optimizer_reports_missing_requirements(self):
        model = LocalGPT(TinyConfig())
        with self.assertRaisesRegex(ValueError, "working CUDA"):
            create_optimizer(model, torch.device("cpu"), "AdamW8bit")
        with patch.dict(sys.modules, {"bitsandbytes": None}):
            with self.assertRaisesRegex(ValueError, "requires bitsandbytes"):
                create_optimizer(model, torch.device("cuda"), "AdamW8bit")
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint optimizer"):
            create_optimizer(model, torch.device("cpu"), "unknown")

    def test_unusable_cuda_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.ones", side_effect=RuntimeError("no kernel image")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(get_training_device(), torch.device("cpu"))

    def test_entrypoint_trains_with_new_and_saved_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_path = Path(directory) / "tokenizer.json"
            argv = ["main.py"]
            with patch.object(parser_module, "ModelConfig", TinyConfig), patch("sys.argv", argv), \
                    patch.object(parser_module, "TRAINING_DIR", Path(directory)), \
                    patch.object(wiki_2_training, "TRAINING_DIR", Path(directory)), \
                    patch.object(wiki_2_training.datasets, "load_dataset", return_value=[
                        {"text": "hello world hello world"},
                        {"text": ""},
                        {"text": "hello world hello world"},
                    ]) as loader, \
                    patch("torch.cuda.is_available", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                main.main()
                argv.append("--resume")
                main.main()
                argv[-1] = "--evaluate-only"
                argv.append("--best")
                main.main()
            self.assertEqual(output.getvalue().count("Starting training..."), 2)
            loader.assert_has_calls([
                call(
                    "parquet", data_files={split: wiki_2_training.TRAINING_URL.replace("/train-", f"/{split}-")},
                    split=split, cache_dir=str(Path(directory) / "cache"),
                    download_config=wiki_2_training.datasets.DownloadConfig(
                        cache_dir=str(Path(directory) / "downloads")),
                ) for split in ("train", "validation")
            ])
            self.assertEqual(loader.call_count, 2)
            state = load_checkpoint(Path(directory) / "checkpoints" / "latest.pt")
            self.assertEqual(state["epoch"], 2)
            self.assertTrue((Path(directory) / "checkpoints" / "best.pt").is_file())
            self.assertEqual((Path(directory) / "train.txt").read_text(encoding="utf-8"),
                             "hello world hello world\n\nhello world hello world\n")
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
        embedding = model.transformer["wte"]
        self.assertIsInstance(embedding, torch.nn.Embedding)
        assert isinstance(embedding, torch.nn.Embedding)
        self.assertIs(model.lm_head.weight, embedding.weight)

    def test_training_updates_weights_and_reaches_all_layers(self):
        torch.manual_seed(0)
        model = LocalGPT(TinyConfig())
        dataset = TokenizedDataset(list(range(10)), TinyConfig.block_size)
        before = model.lm_head.weight.detach().clone()
        with patch("torch.cuda.is_available", return_value=False), \
                patch("model.train.DataLoader", wraps=torch.utils.data.DataLoader) as batches, \
                contextlib.redirect_stdout(io.StringIO()):
            train_local_model(model, dataset, TinyConfig(), epochs=1)
        self.assertFalse(batches.call_args.kwargs["shuffle"])
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            self.assertIsNotNone(gradient, name)
            assert gradient is not None, name
            self.assertTrue(torch.isfinite(gradient).all(), name)

    def test_dataset_targets_and_short_corpus(self):
        dataset = TokenizedDataset([0, 1, 2, 3, 4], 4)
        x, y = dataset[0]
        self.assertEqual(x.tolist(), [0, 1, 2, 3])
        self.assertEqual(y.tolist(), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            TokenizedDataset([0, 1], 4)

    def test_sequential_chunks(self):
        dataset = TokenizedDataset(list(range(12)), block_size=4, stride=4)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0][0].tolist(), [0, 1, 2, 3])
        self.assertEqual(dataset[1][0].tolist(), [4, 5, 6, 7])
        self.assertEqual(dataset[1][1].tolist(), [5, 6, 7, 8])
        with self.assertRaises(IndexError):
            dataset[2]

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        config = TinyConfig()
        config.dropout = 0.2
        config.batch_size = 2
        dataset = TokenizedDataset(list(range(20)), config.block_size)
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            continued = Path(directory) / "continued"
            captured = []

            def capture_checkpoint(path, state):
                save_checkpoint(path, state)
                if state["global_step"] == 2 and not captured:
                    captured.append(load_checkpoint(path))

            torch.manual_seed(123)
            model = LocalGPT(config)
            with patch("torch.cuda.is_available", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch("model.train.save_checkpoint", side_effect=capture_checkpoint):
                    train_local_model(model, dataset, config, epochs=1,
                                      checkpoint_dir=original, save_every=2)
                restored_config = checkpoint_config(captured[0])
                restored = LocalGPT(restored_config)
                train_local_model(restored, dataset, restored_config, epochs=1,
                                  checkpoint_dir=continued, resume=captured[0], save_every=2)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
            original_state = load_checkpoint(original / "latest.pt")
            continued_state = load_checkpoint(continued / "latest.pt")
            self.assertEqual(original_state["global_step"], continued_state["global_step"])
            self.assertEqual(original_state["epoch"], continued_state["epoch"])
            self.assertEqual(original_state["scheduler"], continued_state["scheduler"])
            for key, state in original_state["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, continued_state["optimizer"]["state"][key][name])
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "training data"):
                    train_local_model(restored, TokenizedDataset(list(range(21)), config.block_size),
                                      restored_config, resume=captured[0])

    def test_evaluation_weights_batches_and_restores_mode(self):
        model = LocalGPT(TinyConfig())
        dataset = TokenizedDataset(list(range(10)), TinyConfig.block_size)
        with torch.inference_mode():
            model.eval()
            x, y = zip(*(dataset[i] for i in range(len(dataset))))
            _, expected = model(torch.stack(x), torch.stack(y))
        model.train()
        metrics = evaluate_model(model, dataset, batch_size=4)
        self.assertAlmostEqual(metrics["loss"], expected.item(), places=5)
        self.assertEqual(metrics["tokens"], len(dataset) * TinyConfig.block_size)
        self.assertTrue(model.training)

    def test_scheduler_uses_current_validation_and_resumes_history(self):
        config = TinyConfig()
        dataset = TokenizedDataset(list(range(10)), config.block_size)
        model = LocalGPT(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            losses = [{"loss": value, "perplexity": 100.0} for value in (5.0, 5.1, 5.2)]
            with patch("torch.cuda.is_available", return_value=False), \
                    patch("model.train.evaluate_model", side_effect=losses), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_local_model(model, dataset, config, epochs=3,
                                  checkpoint_dir=directory, validation_dataset=dataset)
            state = load_checkpoint(path)
            self.assertEqual(state["scheduler"]["best"], 5.0)
            self.assertEqual(state["scheduler"]["num_bad_epochs"], 2)
            self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], 6e-4)
            with patch("torch.cuda.is_available", return_value=False), \
                    patch("model.train.evaluate_model", return_value={"loss": 5.3, "perplexity": 100.0}), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_local_model(LocalGPT(config), dataset, config, epochs=1,
                                  checkpoint_dir=directory, validation_dataset=dataset, resume=state)
            state = load_checkpoint(path)
            self.assertEqual(state["scheduler"]["num_bad_epochs"], 0)
            self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], 3e-4)

            # Legacy checkpoints retain their historical best when starting a scheduler.
            state.pop("scheduler")
            with patch("torch.cuda.is_available", return_value=False), \
                    patch("model.train.evaluate_model", return_value={"loss": 5.4, "perplexity": 100.0}), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_local_model(LocalGPT(config), dataset, config, epochs=1,
                                  checkpoint_dir=directory, validation_dataset=dataset, resume=state)
            state = load_checkpoint(path)
            self.assertEqual(state["scheduler"]["best"], 5.0)
            self.assertEqual(state["scheduler"]["num_bad_epochs"], 1)

    def test_generation_crops_context_and_restores_mode(self):
        model = LocalGPT(TinyConfig())
        tokenizer = Mock()
        tokenizer.encode.return_value.ids = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids: ids
        generated = generate_text(model, tokenizer, "prefix", max_new_tokens=6, top_k=1)
        self.assertEqual(len(generated), 11)
        self.assertEqual(generated[:5], [1, 2, 3, 4, 5])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
