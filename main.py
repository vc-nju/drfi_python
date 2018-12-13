import cv2
import numpy as np

from feature_process import Features
from region_detect import Super_Region
from model import RandomForest, MLP, Region2Csv, combine_csv
import time

if __name__ == "__main__":
    # for img_id in range(41,51):
    #     t = time.time()
    #     path = "data/MSRA-B/({}).jpg".format(img_id)
    #     rlist, rmat = Super_Region.get_region(path, 300.)
    #     f = Features(path, rlist, rmat)
    #     csv_path = "data/csv/val/{}.csv".format(img_id)
    #     Region2Csv(f, rlist, 0, csv_path)
    #     print(img_id, time.time() - t)
    # combine_csv()
    RandomForest()
    # MLP()