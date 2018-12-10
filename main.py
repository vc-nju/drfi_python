import cv2
import numpy as np

from region_detect import Super_Region
from feature_process import Features

if __name__ == "__main__":
    img_type = "train"
    img_id = 1
    path = "data/{}_origin/{}.png".format(img_type, img_id)
    rlist, rmat = Super_Region.get_region(path, 300.)
    np.set_printoptions(threshold=np.NaN)
    features = Features(path, rlist, rmat)