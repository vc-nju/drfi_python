'''
import sys
import os
cur_path = os.getcwd()
sys.path.append("../")
from region_detect import super_region,utils
'''
import feature
import pickle
import cv2

if __name__ == '__main__':
    path = '../data/77.jpg'
    imgrgb = cv2.imread(path)
    rlist = None
    region = None
    #rlist,region = super_region.Super_Region.get_region(path,166)
    with open("d", "rb+") as file:
        [rlist, region] = pickle.load(file) 
    print(region.shape)
    reg_feature = feature.RegionFeature(imgrgb,rlist,region)
    print(reg_feature.reg_features)