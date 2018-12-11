'''
@Description: Generate region_features, contrast_features, background features, combine features.
@Author: lizhihao6
@Github: https://github.com/lizhihao6
@Date: 2018-11-26 00:09:40
@LastEditors: lizhihao6
@LastEditTime: 2018-12-11 01:18:39
'''
import cv2
import numpy as np

from .utils import Utils


class Features():
    '''
    @description:  Using region lists and region matrix to calculate super regions features.
                   More details about the defination of features, please see the paper:
                   http://arxiv.org/pdf/1410.5926v1
    @param {img path, region lists, region matrix} 
    @return: Features Class
    '''

    def __init__(self, path, rlist, rmat):
        '''
        @description: The init of feature class 
        @param {img path, region lists, region matrix} 
        @return: None
        '''
        self.rgb = cv2.imread(path)
        self.rlist = rlist.append(Utils.get_background(
            self.rgb.shape[0], self.rgb.shape[1]))
        self.rmat = rmat
        self.utils = Utils(self.rgb, self.rlist, self.rmat)
        self.features29 = self.get_29_features()
        self.reg_features = self.get_region_features()
        self.con_features = self.get_contrast_features()
        self.bkp_features = self.get_background_features()
        self.comb_features = self.get_combine_features()

    def get_region_features(self):
        '''
        @description: Generate region features. 
        @param {None} 
        @return: Region features
        '''
        num_reg = len(self.rlist) - 1
        reg_features = np.zeros([num_reg, 35])
        reg_features[:, 0:6] = self.utils.coord[:-1, 0:6]
        reg_features[:, 6] = self.utils.edge_nums[:-1, 0]
        reg_features[:, 7] = self.utils.coord[:-1, 6]
        reg_features[:, 8:17] = self.utils.color_var[:-1]
        reg_features[:, 17:32] = self.utils.tex_var[:-1]
        reg_features[:, 32] = self.utils.lbp_var[:-1, 0]
        reg_features[:, 33] = self.utils.a[:-1, 0]
        reg_features[:, 34] = self.utils.neigh_areas[:-1, 0]
        return reg_features

    def get_contrast_features(self):
        '''
        @description: Generate contrast features.
        @param {None} 
        @return: Contrast features
        '''
        con_features = np.sum(self.features29, axis=1)[:, :-1]
        con_features = con_features.T
        return con_features

    def get_background_features(self):
        '''
        @description: Generate background features.
        @param {None} 
        @return: Background features
        '''
        bkg_features = self.features29[:, -1, :-1]
        bkg_features = bkg_features.T
        return bkg_features

    def get_combine_features(self):
        '''
        @description: Generate background features.
        @param {type} 
        @return: Combine features lists
                 example: [ np.shape(a, 29), np.shape(b, 29)... ], a means region 0's neighboor regions num.
        '''
        edge_ids = self.utils.edge_neigh
        comb_features = []
        for i in range(len(edge_ids)):
            ids = edge_ids[i]
            features = np.zeros([29+7, len(ids)])
            features[:29, :] = self.features29[:, i, ids]
            features[29:, :] = self.utils.edge_prop[i, ids, :].T
            comb_features.append(features.T)
        return comb_features

    def get_29_features(self):
        '''
        @description: 29-dim features of all super regions and background region.
        @param {None} 
        @return: 29-dim features
        '''
        num_reg = len(self.rlist)
        features = np.zeros([29, num_reg, num_reg])
        dot = self.utils.dot
        for i in range(9):
            features[i] = dot(self.utils.color_avg[:, i])
        features[9] = dot(self.rgb, hist=True)
        features[10] = dot(self.utils.hsv, hist=True)
        features[11] = dot(self.utils.lab, hist=True)
        for i in range(15):
            features[i+12] = dot(self.utils.tex_avg[:, i])
        features[27] = dot(self.utils.tex, hist=True)
        features[28] = dot(self.utils.lbp, hist=True)
        return features
