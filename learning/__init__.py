"""
Learning for training and evaluating the LocalGPT model.
"""

from .parsing import parsers, TrainingSetup
from .data_setup import setup_data, DataSetup



__all__ = ["parsers", "TrainingSetup", "DataSetup", "setup_data"]
