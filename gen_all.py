import numpy as np
import pandas as pd
import os

def combine_csv():
    data = []
    for i in range(1, 500):
        if i%5 != 0:
            path = "data/csv/train/{}.csv".format(i)
            if os.path.exists(path):
                data.append(pd.read_csv(path).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("data/csv/train/all.csv")
    data = []
    for i in range(1, 501):
        if i%5 == 0:
            path = "data/csv/val/{}.csv".format(i)
            if os.path.exists(path):
                data.append(pd.read_csv(path).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("data/csv/val/all.csv")

if __name__ == "__main__":
    combine_csv()