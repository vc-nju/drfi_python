import numpy as np
# from model import RandomForest
from feature_process import Features
from region_detect import Super_Region, Region2Csv

TRAIN_IMGS = 1
C_LIST = [0.2, 0.3, 0.5, 1.0]


class Img_Data:
    def __init__(self, img_path):
        self.img_path = img_path
        self.rlist, self.rmat = Super_Region.get_region(img_path, 100.)
        features = Features(img_path, self.rlist, self.rmat)
        self.comb_features = features.comb_features
        self.rlists = [self.rlist]
        self.rmats = [self.rmat]
        self.feature93s = [features.features93]


if __name__ == "__main__":

    import pickle
    with open("imdata.pkl", "rb+") as file:
        im_data = pickle.load(file)
    with open("simi.pkl", "rb+") as file:
        similarity = pickle.load(file)
    num_reg = len(im_data.rlist)
    # print(num_reg)
    # print(similarity)
    # print(similarity)
    for c in C_LIST:
        rlist, rmat = Super_Region.combine_region(
            similarity, c, im_data.rlist, im_data.rmat)
        print(len(rlist))
        sizes = [len(r[0]) for r in rlist]
        # import matplotlib.pyplot as plt
        # plt.bar(range(len(sizes)), sizes)
        # plt.show(sizes)
        # print(aaa)
    #     im_data.rlists.append(rlist)
    #     im_data.rmats.append(rmat)
    #     features = Features(im_data.img_path, rlist, rmat,
    #                         need_comb_features=False)
    #     im_data.feature93s.append(features.features93)

