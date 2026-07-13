import json

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_dataset(p, n):
    d = f"{n}/{p}"

    def load(prefix):
        return {
            "train": np.load(f"{d}/{prefix}_train.npy", allow_pickle=True),
            "val": np.load(f"{d}/{prefix}_val.npy", allow_pickle=True),
            "test": np.load(f"{d}/{prefix}_test.npy", allow_pickle=True),
        }

    N = None
    C = None

    try:
        N = load("N")
    except:
        pass

    y = load("y")

    try:
        C = load("C")
    except:
        pass

    if C is not None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        C["train"] = enc.fit_transform(C["train"])

        for k in ("val", "test"):
            C[k] = enc.transform(C[k])

    if N is not None and C is not None:
        for k in ("train", "val", "test"):
            N[k] = np.concatenate([N[k], C[k]], axis=1)
        X = N
    elif N is not None:
        X = N
    elif C is not None:
        X = C
    else:
        raise FileNotFoundError(f"Dataset {d} não possui nem arquivos N nem C.")

    with open(f"{d}/info.json") as f:
        info = json.load(f)

    train_val = (
        {k: X[k] for k in ("train", "val")},
        None,
        {k: y[k] for k in ("train", "val")},
    )

    test = (
        {"test": X["test"]},
        None,
        {"test": y["test"]},
    )

    return train_val, test, info
