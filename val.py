import cv2
import numpy as np
import pandas as pd

from model import RandomForest, MLP
from feature_process import Features
from region_detect import Super_Region, Region2Csv

TRAIN_IMGS = 500
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
            similarity[i, ids] = rf.predict(X)[:, 0]
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
    its = [i for i in range(1, TRAIN_IMGS + 1) if i % 5 == 0]
    csv_paths = ["data/csv/val/{}.csv".format(i) for i in its]
    seg_csv_paths = ["data/csv/val/seg{}.csv".format(i) for i in its]
    img_paths = ["data/MSRA-B/{}.jpg".format(i) for i in its]
    seg_paths = ["data/MSRA-B/{}.png".format(i) for i in its]
    img_datas = []
    for i in range(len(its)):
        im_data = Img_Data(img_paths[i])
        Region2Csv.generate_similar_csv(
            im_data.rlist, im_data.comb_features, seg_paths[i], csv_paths[i])
        img_datas.append(im_data)
        print("finished simi {}".format(i))

    val_csv_path = "data/csv/val/all.csv"
    Region2Csv.combine_csv(csv_paths, val_csv_path)
    rf_simi = RandomForest()
    model_path = "data/model/rf_same_region.pkl"
    rf_simi.load_model(model_path)
    rf_simi.test(val_csv_path)

    for i, im_data in enumerate(img_datas):
        im_data.get_multi_segs(rf_simi)
        csv_temp_paths = []
        for j, rlist in enumerate(im_data.rlists):
            temp_path = "data/csv/temp{}.csv".format(j)
            csv_temp_paths.append(temp_path)
            Region2Csv.generate_seg_csv(
                rlist, im_data.feature93s[j], seg_paths[i], temp_path)
        Region2Csv.combine_csv(csv_temp_paths, seg_csv_paths[i])
        print("finished multi seg {}".format(i))

    val_csv_path = "data/csv/val/seg_all.csv"
    Region2Csv.combine_csv(seg_csv_paths, val_csv_path)
    rf_sal = RandomForest()
    model_path = "data/model/rf_salience.pkl"
    rf_sal.load_model(model_path)
    rf_sal.test(val_csv_path)

    mlp = MLP()
    model_path = "data/model/mlp.pkl"
    mlp.load_model(model_path)

    ground_truths = []
    salience_maps = []
    rf_sal_weight = np.zeros(93)
    for i, im_data in enumerate(img_datas):
        segs_num = len(im_data.rlists)
        if segs_num < len(C_LIST)+1:
            continue
        height = im_data.rmat.shape[0]
        width = im_data.rmat.shape[1]
        salience_map = np.zeros([segs_num, height, width])
        for j, rlist in enumerate(im_data.rlists):
            Y = rf_sal.predict(im_data.feature93s[j])[:, 1]
            for k, r in enumerate(rlist):
                salience_map[j][r] = Y[k]
        
            _, _, weights = rf_sal.get_weights(im_data.feature93s[j])
            rf_sal_weight += np.mean(weights, axis=0)[:, 1]
        
        rf_sal_weight /= len(im_data.rlists)

        ground_truth = cv2.imread(seg_paths[i])[:, :, 0]
        ground_truth[ground_truth == 255] = 1
        x = salience_map.reshape([-1, height*width]).T
        salience_maps.append(x)
        ground_truths.append(ground_truth.reshape(-1))

        result = mlp.predict(x).reshape([height, width, 1])
        result[result>0.5] = 255
        result[result<=0.5] = 0
        cv2.imwrite("data/result/{}.png".format(its[i]), result.astype(np.uint8))
        
        print("finish w {}".format(i))

    X_test = np.array(salience_maps)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.array(ground_truths)
    Y_test = np.concatenate(Y_test, axis=0)
    mlp.test(X_test, Y_test)

    df = pd.DataFrame(rf_sal_weight/len(img_datas))
    df.to_csv("data/csv/rf_sal_weight.csv")
