#! ./.venv/bin/python
import itertools
import random

import numpy as np


def get_data_batch():
    """
    Defines a batch-wise generator of data
    """
    yield 0, {}
    yield 1, {}


def get_data():
    return [batch for batch in get_data_batch()]


import flor

flor.flags.NAME = "kaggle-nlp-disasters"
flor.flags.REPLAY = False

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
