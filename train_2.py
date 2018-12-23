import numpy as np
from model import RandomForest
from feature_process import Features
from region_detect import Super_Region, Region2Csv

# for i in range(1, 401):
#     csv_paths = []
#     if i%5 !=0:
#         path = "data/csv/train/{}.csv".format(i)
#         csv_paths.append(path)
#     else:
#         path = "data/csv/val/{}.csv".format(i)
#         csv_paths.append(path)
        
# Region2Csv.combine_csv(csv_paths, "all_93.csv")

import os
import pandas as pd

def combine_csv():
    data = []
    for i in range(1, 400):
        if i%5 != 0:
            path = "data/csv/train/{}.csv".format(i)
            if os.path.exists(path):
                data.append(pd.read_csv(path).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("all_93.csv")
    data = []
    for i in range(1, 401):
        if i%5 == 0:
            path = "data/csv/val/{}.csv".format(i)
            if os.path.exists(path):
                data.append(pd.read_csv(path).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("all_93.csv")

if __name__ == "__main__":
    combine_csv()

# if __name__ == "__main__":
#     train_csv_path = "data/csv/train/all.csv"
#     rf_sal = RandomForest()
#     rf_sal.train(train_csv_path)
#     model_path = "data/model/rf_salience_2.pkl"
#     rf_sal.save_model(model_path)

#     test_csv_path = "data/csv/train/520.csv"
#     rf_sal.test(test_csv_path)