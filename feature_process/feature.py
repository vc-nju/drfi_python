import cv2
import copy
import numpy as np

from .utils import Utils


class Features():
    """Generate img features useing reagions lists and regions mat.

    Use the reagions lists and regions mat to generate 93-dim features and 222-dim features.

    Attributes:
        features93: A 93-dim features used to generate salience map. 
                    Which shape is [Num of regions, 93]
        comb_features(optional):  A 222-dim features used to combine regions.
    """

    def __init__(self, path, rlist, rmat, need_comb_features=True):
        self.rgb = cv2.imread(path)
        self.rlist = rlist
        _list = copy.deepcopy(rlist)
        _list.append(Utils.get_background(
            self.rgb.shape[0], self.rgb.shape[1]))
        self.utils = Utils(self.rgb, _list, rmat, need_comb_features)
        self.features29 = self.get_29_features()
        self.features93 = self.get_features93()
        if need_comb_features:
            self.comb_features = self.get_combine_features()

    def get_features93(self):
        num_reg = len(self.rlist)
        features93 = np.zeros([num_reg, 93])
        features93[:, :35] = self.get_region_features()
        features93[:, 35:35+29] = self.get_contrast_features()
        features93[:, 64:] = self.get_background_features()
        return features93

    def get_region_features(self):
        num_reg = len(self.rlist)
        reg_features = np.zeros([num_reg, 35])
        reg_features[:, 0:6] = self.utils.coord[:-1, 0:6]
        reg_features[:, 6] = self.utils.edge_nums[:-1]
        reg_features[:, 7] = self.utils.coord[:-1, 6]
        reg_features[:, 8:17] = self.utils.color_var[:-1]
        reg_features[:, 17:32] = self.utils.tex_var[:-1]
        reg_features[:, 32] = self.utils.lbp_var[:-1, 0]
        reg_features[:, 33] = self.utils.a[:-1, 0]
        reg_features[:, 34] = self.utils.neigh_areas[:-1]
        return reg_features

    def get_contrast_features(self):
        con_features = np.sum(self.features29, axis=1)[
            :, :-1] / len(self.rlist)
        con_features = con_features.T
        return con_features

    def get_background_features(self):
        bkg_features = self.features29[:, -1, :-1]
        bkg_features = bkg_features.T
        return bkg_features

    def get_combine_features(self):
        edge_ids = self.utils.edge_neigh
        num_reg = len(self.rlist)
        comb_features = [{"i_id": i, "j_ids": [], "features":[]}
                         for i in range(num_reg)]
        for i in range(num_reg):
            ids = edge_ids[i]
            features = np.zeros([222, len(ids)])
            features[:93] = np.repeat(
                self.features93[i], len(ids)).reshape(93, -1)
            features[93:186] = self.features93[ids].T
            features[186:186+29] = self.features29[:, i, ids]
            features[215:] = self.utils.edge_prop[i, ids, :].T
            comb_features[i]["j_ids"] = ids
            comb_features[i]["features"] = features.T
        return comb_features

    def get_29_features(self):
        num_reg = len(self.rlist)
        features = np.zeros([29, num_reg+1, num_reg+1])
        dot = self.utils.dot
        for i in range(9):
            features[i] = dot(self.utils.color_avg[:, i])
        features[9] = dot(self.rgb, hist=True)
        features[10] = dot(self.utils.hsv, hist=True)
        features[11] = dot(self.utils.lab, hist=True)
        for i in range(15):
            features[i+12] = dot(self.utils.tex_avg[:, i])
        features[27] = dot(self.utils.tex, hist=True)
        features[28] = dot(np.int16(self.utils.lbp), hist=True)
        return features
