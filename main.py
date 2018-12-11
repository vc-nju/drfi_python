import cv2
import numpy as np

# import pickle
# _file = open('./rlist.rlist','rb')
# rlist = pickle.load(_file)
# _file.close()
# f = open('./rmat.rmat','rb')
# rmat = pickle.load(f)
# f.close()

#from test_edges import get_edges
from region_detect import Super_Region #generate_coco_data
from feature_process import Features

if __name__ == "__main__":
    img_type = "train"
    img_id = 0
    path = "data/{}_origin/{}.png".format(img_type, img_id)
    rlist, rmat = Super_Region.get_region(path, 300.)
    np.set_printoptions(threshold=np.NaN)
    features = Features(path, rlist, rmat)
    #print(features.reg_features)
    #print(features.con_features)
    #print(features.bkp_features)
    #print(features.comb_features)
'''
    print("reg_feature shape is {}".format(features.reg_features.shape))
    print("con_feature shape is {}".format(features.con_features.shape))
    print("bkp_feature shape is {}".format(features.bkp_features.shape))
    print("comb_feature len is {}".format(len(features.comb_features)))
    print("comb_feature[0] shape is {}".format(features.comb_features[0].shape))
    print("comb_feature[0] shape is {}".format(features.comb_features[0].shape))
'''
