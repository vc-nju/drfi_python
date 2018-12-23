import numpy as np
from model import RandomForest
from feature_process import Features
from region_detect import Super_Region, Region2Csv

TRAIN_IMGS = 1
# C_LIST = [0.2, 0.3, 0.5, 1.0]
C_LIST = [20, 80, 150, 350, 550, 900]

from skimage import io

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
            print(len(rlist))
            disp_mat = (rmat)/(len(rlist) - 1)
            _path = "./threshold_{}.jpg".format(c)
            io.imsave(_path, disp_mat)
            # cv2.imshow('disp_mat', disp_mat)
            # cv2.waitKey(0)
            self.rlists.append(rlist)
            self.rmats.append(rmat)
            features = Features(self.img_path, rlist, rmat,
                                need_comb_features=False)
            self.feature93s.append(features.features93)

if __name__ == "__main__":
    # its = range(1, TRAIN_IMGS+2)
    # csv_paths = ["data/csv/train/{}.csv".format(i) for i in its]
    # img_paths = ["data/MSRA-B/{}.jpg".format(i) for i in its]
    # seg_paths = ["data/MSRA-B/{}.png".format(i) for i in its]
    # csv_paths = ["data/csv/train/520.csv"]
    # img_paths = ["data/MSRA-B/520.jpg"]
    # seg_paths = ["data/MSRA-B/520.png"]
    # img_datas = []
    # for i in range(TRAIN_IMGS):
    #     im_data = Img_Data(img_paths[i])
    #     Region2Csv.generate_similar_csv(
    #         im_data.rlist, im_data.comb_features, seg_paths[i], csv_paths[i])
    #     img_datas.append(im_data)

    csv_paths = []
    img_paths = []
    seg_paths = []
    img_datas = []
    for i in range(1, 401):
        print("img_data", i)
        if i%5 !=0:
            csv_paths.append("data/csv/train/{}.csv".format(i))
        else:
            csv_paths.append("data/csv/val/{}.csv".format(i))
        img_paths.append("data/MSRA-B/{}.jpg".format(i))
        seg_paths.append("data/MSRA-B/{}.png".format(i))
        im_data = Img_Data(img_paths[i-1])
        # Region2Csv.generate_similar_csv(
        #     im_data.rlist, im_data.comb_features, seg_paths[i-1], csv_paths[i-1])
        img_datas.append(im_data)

    #train_csv_path = "data/csv/train/all.csv" #attention: train 103 is 222 dimension
    # Region2Csv.combine_csv(csv_paths, train_csv_path)
    rf_simi = RandomForest()
    # rf_simi.train(train_csv_path)
    # model_path = "data/model/rf_same_region.pkl"
    # rf_simi.save_model(model_path)
    rf_simi.load_model("data/model/rf_same_region.pkl")


    train_csv_path = "data/csv/train/all_93.csv"
    for i, im_data in enumerate(img_datas):
        print("rf_sal number is: ", i)
        im_data.get_multi_segs(rf_simi)
        csv_temp_paths = []
        for j, rlist in enumerate(im_data.rlists):
            temp_path = "data/csv/temp{}.csv".format(j)
            csv_temp_paths.append(temp_path)
            Region2Csv.generate_seg_csv(
                rlist, im_data.feature93s[j], seg_paths[i], temp_path)
        Region2Csv.combine_csv(csv_temp_paths, csv_paths[i])

    Region2Csv.combine_csv(csv_paths, train_csv_path)
    rf_sal = RandomForest()
    rf_sal.train(train_csv_path)
    model_path = "data/model/rf_salience.pkl"
    rf_sal.save_model(model_path)

    # rf_sal.load_model("data/model/rf_salience.pkl")
    # test_path = "data/csv/temp_22.csv"
    # test_im_data = Img_Data("data/MSRA-B/22.jpg")
    # test_seg_path = "data/MSRA-B/22.png"
    # test_im_data.get_multi_segs(rf_simi)
    # for j, rlist in enumerate(test_im_data.rlists):
    #     Region2Csv.generate_seg_csv(
    #         rlist, test_im_data.feature93s[j], test_seg_path, test_path)
    # rf_sal.test(test_path)
