"""TODO(wikitext): Add a description here."""


import os
from pathlib import Path

import datasets


TRAINING_DIR = Path(__file__).resolve().parent / "training" / "wikitext-2"
TRAINING_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/train-00000-of-00001.parquet"
)


def prepare_wikitext2():
    """Prepare the WikiText-2 raw training split inside the repository.

    Use the maintained Hub dataset rather than the legacy archive URLs in the
    original builder below. Validation and test text never enter the corpus.
    """
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = TRAINING_DIR / "train.txt"
    if corpus_path.is_file() and corpus_path.stat().st_size:
        return corpus_path

    training = datasets.load_dataset(
        "parquet",
        data_files={"train": TRAINING_URL},
        split="train",
        cache_dir=str(TRAINING_DIR / "cache"),
        download_config=datasets.DownloadConfig(cache_dir=str(TRAINING_DIR / "downloads")),
    )
    temporary_path = corpus_path.with_suffix(".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as corpus:
            for row in training:
                corpus.write(row["text"].rstrip("\r\n") + "\n")
        temporary_path.replace(corpus_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return corpus_path


_CITATION = """\
@misc{merity2016pointer,
      title={Pointer Sentinel Mixture Models},
      author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
      year={2016},
      eprint={1609.07843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
 The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike
 License.
"""
_HOMEPAGE = "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/"
_LICENSE = "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"
_DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext"


class WikitextConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(self, data_url, **kwargs):
        """BuilderConfig for Wikitext

        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikitextConfig, self).__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url


class Wikitext(datasets.GeneratorBasedBuilder):
    """TODO(wikitext_103): Short description of my dataset."""

    # TODO(wikitext_103): Set up version.
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        WikitextConfig(
            name="wikitext-103-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-v1.zip",
            description="Word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
        WikitextConfig(
            name="wikitext-2-v1",
            data_url=_DATA_URL + "/" + "wikitext-2-v1.zip",
            description="Word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
        WikitextConfig(
            name="wikitext-103-raw-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-raw-v1.zip",
            description="Raw level dataset: the raw tokens before the addition of <unk> tokens. "
            "They should only be used for character level work or for creating newly derived datasets.",
        ),
        WikitextConfig(
            name="wikitext-2-raw-v1",
            data_url=_DATA_URL + "/" + "wikitext-2-raw-v1.zip",
            description="Raw level dataset: the raw tokens before the addition of <unk> tokens. "
            "They should only be used for character level work or for creating newly derived datasets.",
        ),
    ]

    def _info(self):
        # TODO(wikitext): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(wikitext): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        if self.config.name == "wikitext-103-v1":
            data_file = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(data_file, "wikitext-103")
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"data_file": os.path.join(data_dir, "wiki.test.tokens"), "split": "test"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"data_file": os.path.join(data_dir, "wiki.train.tokens"), "split": "train"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"data_file": os.path.join(data_dir, "wiki.valid.tokens"), "split": "valid"},
                ),
            ]
        else:
            if self.config.name == "wikitext-103-raw-v1":
                data_file = dl_manager.download_and_extract(self.config.data_url)
                data_dir = os.path.join(data_file, "wikitext-103-raw")
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={"data_file": os.path.join(data_dir, "wiki.test.raw"), "split": "test"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"data_file": os.path.join(data_dir, "wiki.train.raw"), "split": "train"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"data_file": os.path.join(data_dir, "wiki.valid.raw"), "split": "valid"},
                    ),
                ]
            else:
                if self.config.name == "wikitext-2-raw-v1":
                    data_file = dl_manager.download_and_extract(self.config.data_url)
                    data_dir = os.path.join(data_file, "wikitext-2-raw")
                    return [
                        datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={"data_file": os.path.join(data_dir, "wiki.test.raw"), "split": "test"},
                        ),
                        datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={"data_file": os.path.join(data_dir, "wiki.train.raw"), "split": "train"},
                        ),
                        datasets.SplitGenerator(
                            name=datasets.Split.VALIDATION,
                            gen_kwargs={"data_file": os.path.join(data_dir, "wiki.valid.raw"), "split": "valid"},
                        ),
                    ]
                else:
                    if self.config.name == "wikitext-2-v1":
                        data_file = dl_manager.download_and_extract(self.config.data_url)
                        data_dir = os.path.join(data_file, "wikitext-2")
                        return [
                            datasets.SplitGenerator(
                                name=datasets.Split.TEST,
                                gen_kwargs={"data_file": os.path.join(data_dir, "wiki.test.tokens"), "split": "test"},
                            ),
                            datasets.SplitGenerator(
                                name=datasets.Split.TRAIN,
                                gen_kwargs={
                                    "data_file": os.path.join(data_dir, "wiki.train.tokens"),
                                    "split": "train",
                                },
                            ),
                            datasets.SplitGenerator(
                                name=datasets.Split.VALIDATION,
                                gen_kwargs={
                                    "data_file": os.path.join(data_dir, "wiki.valid.tokens"),
                                    "split": "valid",
                                },
                            ),
                        ]

    def _generate_examples(self, data_file, split):

        """Yields examples."""
        # TODO(wikitext): Yields (key, example) tuples from the dataset
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                if row.strip():
                    yield idx, {"text": row}
                else:
                    yield idx, {"text": ""}
