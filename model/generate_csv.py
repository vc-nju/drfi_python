import os
import cv2
import numpy as np
import pandas as pd


class Region2Csv():

    def __init__(self, features, rlist, img_id, csv_path):
        self.features = features
        self.generate_csv(features, rlist, img_id, csv_path)

   
    def generate_csv(self, features, rlist, img_id, csv_path):
        data = []
        neigh = features.utils.edge_neigh
        for i in range(len(rlist)):
            region_i_id = i
            for j in range(len(neigh[i])):
                region_j_id = neigh[i][j]
                line_data = self.generate_line_data(img_id, region_i_id, region_j_id, 0. )
                data.append(line_data)
            line_data = self.generate_line_data(img_id, region_i_id, region_i_id, 1. )
            data.append(line_data)
        data = np.concatenate(data)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, header=None, index=0)

    def generate_line_data(self, img_id, region_i_id, region_j_id, is_same_region):
        """
        Each line of CSV will be like this:
            | img_id | region_id i | region_id j | is same_region | 222-dim features |
        """
        line_data = np.zeros([1, 4 + 211])
        line_data = np.zeros_like(line_data)
        line_data[0, 0] = img_id
        line_data[0, 1] = region_i_id
        line_data[0, 2] = region_j_id
        line_data[0, 3] = is_same_region
        line_data[0, 4:4+33] = self.features.reg_features[region_i_id]
        line_data[0, 37:37+29] = self.features.con_features[region_i_id]
        line_data[0, 66:66+29] = self.features.bkp_features[region_i_id]
        line_data[0, 95:95+33] = self.features.reg_features[region_j_id]
        line_data[0, 128:128+29] = self.features.con_features[region_j_id]
        line_data[0, 157:157+29] = self.features.bkp_features[region_j_id]
        line_data[0, 186:186+29] = self.features.features29[:, region_i_id, region_j_id].T
        return line_data

def combine_csv():
    data = []
    for i in range(450):
        path = "data/csv/train/{}.csv".format(i)
        if os.path.exists(path):
            data.append(pd.read_csv(path, header=0).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("data/csv/train/all.csv")
    data = []
    for i in range(50):
        path = "data/csv/val/{}.csv".format(i)
        if os.path.exists(path):
            data.append(pd.read_csv(path, header=0).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("data/csv/val/all.csv")