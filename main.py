import cv2
import numpy as np

from feature_process import Features
from region_detect import Super_Region, Region2Csv

import time

def save_csv(img_type, img_id):
    path = "data/{}_origin/{}.png".format(img_type, img_id)
    t1 = time.time()
    rlist, rmat = Super_Region.get_region(path, 100.)
    t2 = time.time()
    features = Features(path, rlist, rmat)
    t3 = time.time()
    Region2Csv(features, rlist,img_type,img_id)
    t4 = time.time()
    print(t2 - t1)
    print(t3 - t2)
    print(t4 - t3)



if __name__ == "__main__":
    for i in range(13,14):
        print("train", i)
        t = time.time()
        save_csv("train", i)
        print(time.time() - t)
    # for i in range(50):
    #     print("val", i)
    #     t = time.time()
    #     save_csv("val", i)
    #     print(time.time() - t)