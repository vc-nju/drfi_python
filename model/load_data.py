import os
import pickle
import numpy as np
import pandas as pd

def _load_data():
    df = pd.read_csv("data/csv/train/all.csv")
    index = [str(i) for i in range(5, 227)]
    X = df[index].values
    Y = df[["4"]].values
    X_p = []
    X_n = []
    for i in range(len(X)):
        _x = X[i, :]
        _x = _x[np.newaxis, :]
        if int(Y[i, 0]) == 1:
            X_p.append(_x)
        else:
            X_n.append(_x)
    neg_num = len(X_n)
    step = len(X_p)//neg_num - 1
    _X_p = [X_p[i] for i in range(0, neg_num*step, step)]
    itr = range(neg_num)
    X_train = []
    Y_train = []
    for i in range(len(X_n)):
        X_train.append(_X_p[i])
        X_train.append(X_n[i])
        Y_train.append(1.)
        Y_train.append(0.)
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.array(Y_train)
    df = pd.read_csv("data/csv/val/all.csv")
    index = [str(i) for i in range(5, 227)]
    X_test = df[index].values
    Y_test = df[["4"]].values[:, 0]
    return X_train, Y_train, X_test, Y_test


def load_data():
    path = "data/csv/data.pkl"
    if os.path.exists(path):
        with open(path, "rb+") as file:
            [X_train, Y_train, X_test, Y_test] = pickle.load(file)
    else:
        X_train, Y_train, X_test, Y_test = _load_data()
        with open(path, "wb+") as file:
            pickle.dump([X_train, Y_train, X_test, Y_test], file)

    return X_train, Y_train, X_test, Y_test

