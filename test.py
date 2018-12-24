import cv2
import numpy as np

from model import RandomForest, MLP
from feature_process import Features
from region_detect import Super_Region, Region2Csv

C_LIST = [20, 80, 350, 900]


class Img_Data:
    def __init__(self, img_path):
        self.img_path = img_path
        self.rlist, self.rmat = Super_Region.get_region(img_path, 100.)
        features = Features(img_path, self.rlist, self.rmat)
        self.comb_features = features.comb_features
        self.rlists = [self.rlist]
        self.rmats = [self.rmat]
        self.feature93s = [features.features93]

    def get_multi_segs(self, rf):
        num_reg = len(self.rlist)
        similarity = np.ones([num_reg, num_reg])
        for i in range(num_reg):
            ids = self.comb_features[i]["j_ids"]
            X = self.comb_features[i]["features"]
            similarity[i, ids] = 1-rf.predict(X)[:, 1]
        for c in C_LIST:
            rlist, rmat = Super_Region.combine_region(
                similarity, c, self.rlist, self.rmat)
            if len(rlist) == 1:
                continue
            self.rlists.append(rlist)
            self.rmats.append(rmat)
            features = Features(self.img_path, rlist, rmat,
                                need_comb_features=False)
            self.feature93s.append(features.features93)


if __name__ == "__main__":
    img_id = 1036
    img_path = "data/MSRA-B/{}.jpg".format(img_id)
    im_data = Img_Data(img_path)

    rf_simi = RandomForest()
    model_path = "data/model/rf_same_region.pkl"
    rf_simi.load_model(model_path)
    rf_sal = RandomForest()
    model_path = "data/model/rf_salience.pkl"
    rf_sal.load_model(model_path)

    im_data.get_multi_segs(rf_simi)
    segs_num = len(im_data.rlists)
    height = im_data.rmat.shape[0]
    width = im_data.rmat.shape[1]
    salience_map = np.zeros([segs_num, height, width])
    for i, rlist in enumerate(im_data.rlists):
        Y = rf_sal.predict(im_data.feature93s[i])[:, 1]
        for j, r in enumerate(rlist):
            salience_map[i][r] = Y[j]
    X_test = salience_map.reshape([-1, height*width]).T

    mlp = MLP()
    model_path = "data/model/mlp.pkl"
    mlp.load_model(model_path)
    Y = mlp.predict(X_test).reshape([height, width])*255

    img = np.zeros([height, width*2, 3], dtype=np.uint8)
    img[:, :width, :] = cv2.imread(img_path)
    img[:, width:, :] = Y.repeat(3).reshape([height, width, 3])
    print("finished~( •̀ ω •́ )y")
    cv2.imshow("result", img)
    cv2.waitKey(0)
