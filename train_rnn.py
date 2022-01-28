import pandas as pd 
import numpy as np 
import torch   
from torchtext import data  

SEED = 2022
torch.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)
fields = [('words',TEXT),('target', LABEL)]
