from region_detect import Super_Region
import pickle
rlist = None
region = None
with open("d", "rb+") as file:
    [rlist, region] = pickle.load(file) 

from feature_process.feature import Features
import cv2
im = cv2.imread("data/77.jpg")
rf = Features(im, rlist, region)
rf.get_region_features()