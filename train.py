#! ./.venv/bin/python

import itertools

import flor
import numpy as np

print("flor imported")


flor.flags.NAME = "kaggle-nlp-disasters"
flor.flags.REPLAY = False
"""
INDEX   path
MODE    weak | strong
PID     k,n
EPSILON 0.xx
"""
print("flor flags set")


def get_dna_kmers():
    vocab = ["A", "T", "C", "G"]
    packet = {"X": None, "__record__": []}
    for kmer in itertools.permutations(vocab):
        packet["X"] = "".join(kmer)
        packet["__record__"].append(packet["X"])
        yield packet


training_data = get_dna_kmers()
history = None
print("loaded training data")


"""
The main loop of deep learning
cycles over all of the data each epoch
"""
for e in flor.it(range(100)):
    print(f"ENTERING: {e}")
    if flor.SkipBlock.step_into("batchwise-loop"):
        for s, batch in enumerate(training_data):
            """
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item()}")
            """
            if s % 10 == 0:
                print(s, batch["X"])
            history = batch["__record__"]
    flor.SkipBlock.end(history)
    print(f"EXITING: {e}")
