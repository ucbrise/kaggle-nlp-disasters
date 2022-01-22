#! ./.venv/bin/python
import itertools
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import flor

flor.flags.NAME = "kaggle-nlp-disasters"
flor.flags.REPLAY = False

DEEP_LEARNING = False


def get_data(mode="pandas", test=False):
    if mode == "pandas":
        return pd.read_csv("data/train.csv" if not test else "data/test.csv")
    else:
        raise RuntimeError(f"Mode {mode} not implemented in function `get_data`")


data = get_data()
if not DEEP_LEARNING:
    # try scikit learn first
    def key_mapper(s: str, _=[]):
        if s not in _:
            k = len(_)
            _.append(k)
            return k + 1
        else:
            return _.index(s) + 1

    integerMapped_keywords = (
        data["keyword"].apply(lambda k: key_mapper(k) if not k is np.NaN else 0)
    ).array.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        integerMapped_keywords, data["target"]
    )
    for trial in flor.it(range(1)):
        # Naive first step: classify based on keyword
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=1,
            max_depth=3,
            random_state=flor.pin("clf_seed", random.randint(0, 5000)),
        ).fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        print("Validation score: ", score)

        test_data = get_data(test=True)
        integerMapped_test = (
            test_data["keyword"]
            .apply(lambda k: key_mapper(k) if not k is np.NaN else 0)
            .array.reshape(-1, 1)
        )
        preds = clf.predict(integerMapped_test)
        # print("Test score: ", tscore)

        preds_df = pd.DataFrame({"id": test_data["id"], "target": preds})
        preds_df.to_csv(f"data/output_{trial}.csv", index=False)

else:
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

# print("FINISHED")
