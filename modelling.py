from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
from textwrap import wrap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 160
BATCH_SIZE = 16

pre_trained_model = 'bert-base-uncased'

# building a head classifier to add to the body(our pretrained model)
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model, return_dict=False, from_tf=True) # body
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
    