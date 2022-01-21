#! ./.venv/bin/python
import itertools
import random

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
    step = 0
    vocab = ["A", "T", "C", "G"]
    packet = {"X": None, "__record__": []}
    while step != len(vocab) ** 4:
        kmer = [vocab[random.randint(0, len(vocab) - 1)] for _ in range(4)]
        packet["X"] = "".join(kmer)
        if packet["X"] not in packet["__record__"]:
            packet["__record__"].append(packet["X"])
            yield step, packet
            step += 1


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
        for s, batch in training_data:
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
    training_data = get_dna_kmers()
    print(f"EXITING: {e}")
