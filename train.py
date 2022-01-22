#! ./.venv/bin/python
import itertools
import random

import numpy as np
import pandas as pd

import flor

flor.flags.NAME = "kaggle-nlp-disasters"
flor.flags.REPLAY = False

"""
TRY SCIKIT LEARN FIRST
WRITE THE README WITH INSTRUCTIONS, KAGGLE
"""

def get_data_batch():
    """
    Defines a batch-wise generator of data
    """
    yield 0, {}
    yield 1, {}


def get_data(mode='pandas'):
    if mode == 'pandas':
        return pd.read_csv('data/train.csv')
    else:
        raise RuntimeError(f'Mode {mode} not implemented in function `get_data`')
data = get_data()


"""
The main loop of deep learning
cycles over all of the data each epoch
"""
model = []
for e in flor.it(range(100)):
    model.append({})
    if flor.SkipBlock.step_into("batchwise-loop"):
        for s, batch in enumerate(get_data()):
            model[-1]["batch"] = batch
            print(s, batch)
    flor.SkipBlock.end(model)
    print(f"END OF EPOCH {e}")
