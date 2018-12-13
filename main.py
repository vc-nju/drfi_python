import cv2
import numpy as np

from feature_process import Features
from region_detect import Super_Region
from model import RandomForest, MLP, Region2Csv
import time

if __name__ == "__main__":
    path = "data/train_origin/0.png"
    t = time.time()
    rlist, rmat = Super_Region.get_region(path, 300.)
    print(time.time() - t)
    t = time.time()
    f = Features(path, rlist, rmat)
    print(time.time() - t)
    t = time.time()
    csv_path = "data/csv/0.csv"
    Region2Csv(f, rlist, 0, csv_path)
    print(time.time() - t)
    t = time.time()