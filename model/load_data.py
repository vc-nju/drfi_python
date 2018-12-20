import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import resample

LABEL_INDEX = 0
FEATURE_INDEX_MIN = 1


def rebalance(X, Y):
    pos_num = np.sum(Y)
    neg_num = len(Y) - pos_num
    X_more = X[Y == 1]
    X_less = X[Y == 0]
    Y_more = np.ones(pos_num)
    Y_less = np.zeros(neg_num)
    if pos_num < neg_num:
        X_more, X_less = X_less, X_more
        Y_more, Y_less = Y_less, Y_more
    X_more = resample(X_more, n_samples=len(X_less), random_state=0)
    Y_more = resample(Y_more, n_samples=len(Y_less), random_state=0)
    X = np.concatenate([X_more, X_less], axis=0)
    X = np.concatenate([Y_more, X_more], axis=0)
    return X, Y


def _load_data(csv_path, rebalance=True):
    df = pd.read_csv(csv_path)
    index = [str(i) for i in range(FEATURE_INDEX_MIN, FEATURE_INDEX_MIN+222)]
    _index = [str(i) for i in range(FEATURE_INDEX_MIN, FEATURE_INDEX_MIN+93)]
    try:
        X = df[index].values
    except:
        X = df[_index].values
    Y = df[[str(LABEL_INDEX)]].values[:, 0]
    if rebalance:
        X, Y = rebalance(X, Y)
    return X, Y


def load_data(csv_path, rebalance=True):
    path = csv_path + ".pkl"
    if os.path.exists(path):
        with open(path, "rb+") as file:
            [X, Y] = pickle.load(file)
    else:
        X, Y = _load_data(csv_path, rebalance)
        with open(path, "wb+") as file:
            pickle.dump([X, Y], file)
    return X, Y
