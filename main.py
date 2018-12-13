import cv2
import numpy as np

from feature_process import Features
from region_detect import Super_Region, Region2Csv, combine_csv
from model import RandomForest, MLP

if __name__ == "__main__":
    rf = RandomForest()