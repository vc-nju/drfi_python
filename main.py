import cv2

from region_detect import Super_Region, generate_coco_data
from feature_process import Features


if __name__ == "__main__":
    # generate_coco_data()
    img_type = "train"
    img_id = 0
    path = "data/{}_origin/{}.png".format(img_type, img_id)
    rlist, rmat = Super_Region.get_region(path, 300.)
    print(len(rlist))
    features = Features(path, rlist, rmat)
    print("reg_feature shape is {}".format(features.reg_features.shape))
    print("con_feature shape is {}".format(features.con_features.shape))
    print("bkp_feature shape is {}".format(features.bkp_features.shape))
    print("comb_feature len is {}".format(len(features.comb_features)))
<<<<<<< HEAD
    print("comb_feature[0] shape is {}".format(features.comb_features[0].shape))
=======
    print("comb_feature[0] shape is {}".format(
        features.comb_features[0].shape))
>>>>>>> 330f80ff16ab4f7858a455adc1e45f9773513fba
