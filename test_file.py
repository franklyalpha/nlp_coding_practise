# should only be input to python console

from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import DATASETS

import torch
import torch.nn as nn
import os  # for file loading procedures
from pathlib import Path, PureWindowsPath
from llm.prepare_text_data import *

dataset = torch.load("llm/dataset_instance")
vocab = torch.load("llm/vocab_obj")
