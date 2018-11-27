import numpy as np
import struct
import cv2
from utils import Utils


class RegionFeature():
    def __init__(self, rgb, rlist, rmat):
        self.rgb = rgb
        self.rlist = rlist
        self.rmat = rmat
        self.utils = Utils(rgb, rlist, rmat)
        self.reg_features = self.get_region_features()

    def get_region_features(self): #property descriptor
        num_reg = len(self.rlist)
        reg_features = np.zeros([num_reg, 35])
        reg_features[:, 0:6] = self.utils.coord[0:6]
        reg_features[:, 6] = self.utils.edge_nums
        reg_features[:, 7] = self.utils.coord[7]
        reg_features[:, 8:17] = self.utils.color_var
        reg_features[:, 17:32] = self.utils.tex_var
        reg_features[:, 32] = self.utils.lbp_var
        reg_features[:, 33] = self.utils.a
        reg_features[:, 34] = self.utils.neigh_areas
        #reg_min = np.min(reg_features)
        #reg_max = np.max(reg_features)
        #reg_features = (reg_features - reg_min) / (reg_max - reg_min)
        return reg_features

    def get_contrast_features(self):
        num_reg = len(self.rlist)
        con_features = np.zeros([num_reg, 29])
        dot = self.utils.dot
        for i in range(9):
            con_features[:, i] = dot(self.utils.color_avg[:, i])
        con_features[:, 9] = dot(self.rgb, hist=True)
        con_features[:, 10] = dot(self.utils.hsv, hist=True)
        con_features[:, 11] = dot(self.utils.lab, hist=True)
        for i in range(15):
            con_features[:, i+12] = dot(self.utils.tex_avg[:, i])
        con_features[:, 27] = dot(self.utils.tex, hist=True)
        con_features[:, 11] = dot(self.utils.lbp, hist=True)
        return con_features

    def get_background_features(self):
        num_reg = len(self.rlist)
        bkg_features = np.zeros([num_reg, 29])
        utils = Utils(self.rgb, self.rlist+self.utils.blist, self.rmat)
        dot = utils.dot(self.rgb, bkg=True)
        for i in range(9):
            bkg_features[:, i] = dot(utils.color_avg[:, i])
        bkg_features[:, 9] = dot(self.rgb, hist=True)
        bkg_features[:, 10] = dot(utils.hsv, hist=True)
        bkg_features[:, 11] = dot(utils.lab, hist=True)
        for i in range(15):
            bkg_features[:, i+12] = dot(utils.tex_avg[:, i])
        bkg_features[:, 27] = dot(utils.tex, hist=True)
        bkg_features[:, 11] = dot(utils.lbp, hist=True)
        return bkg_features
