import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
fields = [('words', text_field), ('target', label_field)]

train, valid = TabularDataset.splits(path = 'data', train='train_rnn.csv', validation='valid_rnn.csv',format='CSV', fields=fields, skip_header=True)

train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text), sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text), sort=True, sort_within_batch=True)

text_field.build_vocab(train, min_freq=3)