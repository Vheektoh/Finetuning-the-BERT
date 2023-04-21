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

from transformers import DataCollatorWithPadding

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 160
BATCH_SIZE = 16

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len, include_raw_text=False):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_raw_text = include_raw_text
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
        review,
        max_length=self.max_len,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt'
        )

        output = {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'targets' : torch.tensor(target, dtype=torch.long)
            }
        
        if self.include_raw_text:
            output['review_text'] = review
        
        return output


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding= 'longest')
    
def create_data_loader(df, tokenizer, max_len=MAX_LEN,batch_size=BATCH_SIZE, include_raw_text=False):
    ds = ReviewDataset(
        reviews = df['Review Text'].to_list(),
        targets = df['Rating'].to_list(),
        tokenizer= tokenizer,
        max_len= max_len,
        include_raw_text=include_raw_text
    )
    return DataLoader(ds, batch_size=batch_size, collate_fn= collator)


