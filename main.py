import cv2
import numpy as np

from feature_process import Features
from region_detect import Super_Region, Region2Csv, combine_csv
from model import RandomForest, MLP

def generate_csv(img_type, img_id):
    """
    img_id must be int
    """
    img_path = "data/MSRA-B/{}.jpg".format(img_id)
    seg_path = "data/MSRA-B/{}.png".format(img_id)
    rlist, rmat = Super_Region.get_region(img_path, 100.)
    features = Features(img_path, rlist, rmat)
    Region2Csv(features, rlist, img_type, img_id, seg_path)

if __name__ == "__main__":
    for i in range(40):
        generate_csv("train", i)
    for j in range(40, 50):
        generate_csv("val", j)