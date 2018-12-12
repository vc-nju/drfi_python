import cv2
import numpy as np

from feature_process import Features
from region_detect import Super_Region, Region2Csv

def save_csv(img_type, img_id):
    path = "data/{}_origin/{}.png".format(img_type, img_id)
    rlist, rmat = Super_Region.get_region(path, 100.)
    features = Features(path, rlist, rmat)
    Region2Csv(features, rlist,img_type,img_id)


if __name__ == "__main__":
    for i in range(82,450):
        print("train", i)
        save_csv("train", i)
    for i in range(50):
        print("val", i)
        save_csv("val", i)