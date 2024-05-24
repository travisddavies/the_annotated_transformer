import os
from os.path import exists
import torch
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torchtext.optim.lr_scheduler import to_map_style_dataset
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import roch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore')
RUN_EXAMPLES = True
