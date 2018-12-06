import cv2
import numpy as np

from .utils import Utils


class Features():
    def __init__(self, path, rlist, rmat):
        self.rgb = cv2.imread(path)
        self.rlist = rlist
        self.rmat = rmat
        self.utils = Utils(self.rgb, rlist, rmat)
        self.features29 = self.get_29_features()
        self.reg_features = self.get_region_features()
        self.con_features = self.get_contrast_features()
        self.bkp_features = self.get_background_features()
        self.comb_features = self.get_combine_features()

    def get_region_features(self):
        num_reg = len(self.rlist)
        reg_features = np.zeros([35, num_reg])
        reg_features[0:6] = self.utils.coord[:, 0:6]
        reg_features[6] = self.utils.edge_nums[:, 0]
        reg_features[7] = self.utils.coord[:, 6]
        reg_features[8:17] = self.utils.color_var
        reg_features[17:32] = self.utils.tex_var
        reg_features[32] = self.utils.lbp_var
        reg_features[33] = self.utils.a[:, 0]
        reg_features[34] = self.utils.neigh_areas[:, 0]
        reg_features = reg_features.T
        return reg_features

    def get_contrast_features(self):
        con_features = np.sum(self.features29, axis=1)[:, 0, :-1]
        con_features = con_features.T
        return con_features

    def get_background_features(self):
        bkg_features = self.features29[:, -1, :-1]
        bkg_features = bkg_features.T
        return bkg_features

    def get_combine_features(self):
        edge_ids = self.utils.edge_neigh
        comb_features = []
        for i in range(len(edge_ids)):
            ids = edge_ids[i]
            features = np.zeros(29+7, len(ids))
            features[:29, :] = self.features29[:, i, ids]
            features[29:, :] = self.utils.edge_prop[i, ids, :].T
            comb_features.append(features.T)
        return comb_features

    def get_29_features(self):
        num_reg = len(self.rlist)
        features = np.zeros([29, num_reg+1, num_reg+1])
        utils = Utils(self.rgb, self.rlist+self.utils.blist, self.rmat)
        dot = utils.dot
        for i in range(9):
            features[i] = dot(utils.color_avg[:, i])
        features[9] = dot(self.rgb, hist=True)
        features[10] = dot(utils.hsv, hist=True)
        features[11] = dot(utils.lab, hist=True)
        for i in range(15):
            features[i+12] = dot(utils.tex_avg[:, i])
        features[27] = dot(utils.tex, hist=True)
        features[28] = dot(utils.lbp, hist=True)
        return features
