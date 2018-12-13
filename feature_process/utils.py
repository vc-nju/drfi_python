'''
@Description: Utils for generate super regions' features.
@Author: lizhihao6
@Github: https://github.com/lizhihao6
@Date: 2018-11-28 17:02:30
@LastEditors: lizhihao6
@LastEditTime: 2018-12-11 18:11:56
'''

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .LM_filters import makeLMfilters

RATIO_C = 0.2
A_C = 50.
NEIGH_AREAS_C = 0.1
EDGE_NEIGH = 1000

class Utils():
    '''
    @description: Utils class using to geneate features.
    @param {img rgb, region lists, region matrix} 
    @return: Utils class
    '''

    def __init__(self, rgb, rlist, rmat):
        '''
        @description: The init of Utils class.
        @param {img rgb, region lists, region matrix} 
        @return: None
        '''
        self.height, self.width = rmat.shape
        self.rgb, self.rlist, self.rmat = rgb, rlist, rmat
        self.lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        self.hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        self.tex = self.get_tex()
        self.lbp = self.get_lbp()
        self.coord = self.get_coord()
        imgchan = np.concatenate([self.rgb, self.lab, self.hsv], axis=2)
        self.color_avg, self.color_var = self.get_avg_var(imgchan)
        self.tex_avg, self.tex_var = self.get_avg_var(self.tex)
        self.lbp_avg, self.lbp_var = self.get_avg_var(self.lbp)
        self.edge_neigh = self.get_edges()
        self.w = self.get_w()
        self.a = self.get_a()

    def get_tex(self):
        num_reg = len(self.rlist)
        ml_fiters = self.ml_kernal()
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float) / 255.0
        tex = np.zeros([gray.shape[0], gray.shape[1], 15])
        for i in range(15):
            tex[:, :, i] = cv2.filter2D(gray, cv2.CV_64F, ml_fiters[:, :, i])
        for i in range(15):
            tex_max = np.max(tex[:, :, i])
            tex_min = np.min(tex[:, :, i])
            tex[:, :, i] = (tex[:, :, i] - tex_min)/(tex_max - tex_min) * 255
        tex = tex.astype(np.int32)
        return tex

    def get_lbp(self):
        num_reg = len(self.rlist)
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1.).astype(np.int32)
        _lbp = np.zeros((lbp.shape[0], lbp.shape[1], 1))
        _lbp[:, :, 0] = lbp
        return _lbp

    def get_coord(self):
        '''
        @description: Get postion features of regions.
        @param {None} 
        @return: Coordinary features
        '''
        num_reg = len(self.rlist)
        coord = np.zeros([num_reg, 7])
        EPS = 1.
        for i in range(num_reg):
            sum_y_x = np.sum(np.array(self.rlist[i]), axis=1)
            num_pix = len(self.rlist[i][0])
            coord[i][0:2] = sum_y_x // num_pix
            coord[i][0] /= self.height
            coord[i][1] /= self.width
            sortbyy = sorted(self.rlist[i][0])
            sortbyx = sorted(self.rlist[i][1])
            tenth = int(num_pix*0.1)
            ninetith = int(num_pix*0.9)
            coord[i][2:4] = [sortbyy[tenth] /
                             self.height, sortbyx[tenth]/self.width]
            coord[i][4:6] = [sortbyy[ninetith] /
                             self.height, sortbyx[ninetith]/self.width]
            a = float(sortbyy[-1] - sortbyy[0])
            b = float(sortbyx[-1] - sortbyx[0] + EPS)
            ratio = a/b
            coord[i][6] = ratio * RATIO_C
        return coord

    def get_avg_var(self, a):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, a.shape[2]])
        var = np.zeros([num_reg, a.shape[2]])
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            for j in range(a.shape[2]):
                avg[i, j] = np.sum(a[:, :, j][self.rlist[i]]) / num_pix
                var[i, j] = np.sum(
                    (a[:, :, j][self.rlist[i]] - avg[i, j])**2) / num_pix
        var /= 255.**2
        return avg, var

    def get_edges(self):
        '''
        @description: 
        @param {None} 
        @return: edge_neigh: the neighbor region of Ri, [(R1,R2,R3...), (R1,R2,R3...),]
        '''
        rmat = self.rmat
        rlist = self.rlist
        shape = (rmat.shape[0], rmat.shape[1], 8, )
        edge_mat = np.zeros(shape, dtype=np.int32)
        edge_mat[:-1, :, 0] += (rmat[1:, :] - rmat[:-1, :])
        edge_mat[1:, :, 1] += (rmat[:-1, :] - rmat[1:, :])
        edge_mat[:, :-1, 2] += (rmat[:, 1:] - rmat[:, :-1])
        edge_mat[:, 1:, 3] += (rmat[:, :-1] - rmat[:, 1:])
        edge_mat[:-1, :-1, 4] += (rmat[1:, 1:] - rmat[:-1, :-1])
        edge_mat[:-1, 1:, 5] += (rmat[1:, :-1] - rmat[:-1, 1:])
        edge_mat[1:, :-1, 6] += (rmat[:-1, 1:] - rmat[1:, :-1])
        edge_mat[1:, 1:, 7] += (rmat[:-1, :-1] - rmat[1:, 1:])
        edge_mat[edge_mat != 0] = 1

        def generate_y_list(y): return [y+1, y-1, y, y, y+1, y+1, y-1, y-1]

        def generate_x_list(x): return [x, x, x+1, x-1, x+1, x-1, x+1, x-1]
        shape = (rmat.shape[0], rmat.shape[1], 2, 8)
        y_x = np.zeros(shape, dtype=np.int32)
        for y in range(shape[0]):
            for x in range(shape[1]):
                y_x[y, x, 0, :] = generate_y_list(y)
                y_x[y, x, 1, :] = generate_x_list(x)
        edge_neigh = []

        for region in rlist:
            neigh_id = []
            for y, x in zip(region[0], region[1]):
                mask = (edge_mat[y, x] != 0)
                y_ = y_x[y, x, 0][mask]
                x_ = y_x[y, x, 1][mask]
                neigh_id += list(set(neigh_id))
            edge_neigh.append(list(set(neigh_id)))

        return edge_neigh

    def get_w(self):
        num_reg = len(self.rlist)
        pos = np.zeros([num_reg, 2])
        for i in range(num_reg):
            reg_array = np.array(self.rlist[i])

            pos[i, :] = np.sum(reg_array, axis=1) / reg_array.shape[1]
        pos[:, 0] /= self.height
        pos[:, 1] /= self.width
        diff = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            diff[i] = np.sum((pos[i,0:2] - pos[:,0:2])**2,axis=1)
        w = np.exp(-1. * diff / 2)
        return w

    def get_a(self):
        a = np.zeros([len(self.rlist), 1])
        a[:, 0] = [float(len(r[0]))/float(self.width*self.height)
                   for r in self.rlist]
        a = np.array(a)*A_C
        return a

    @staticmethod
    def get_background(height, width):
        blist = [(), ()]
        y = [y_ for y_ in range(15)] + \
            [y_ for y_ in range(height-15, height)]
        x = [x_ for x_ in range(width)]
        for y_ in y:
            for x_ in x:
                blist[0] += (y_,)
                blist[1] += (x_,)
        y = [y_ for y_ in range(15, height - 15)]
        x = [x_ for x_ in range(15)] + \
            [x_ for x_ in range(width-15, width)]
        for y_ in y:
            for x_ in x:
                blist[0] += (y_,)
                blist[1] += (x_,)
        return tuple(blist)

    def ml_kernal(self):
        ml_filters = makeLMfilters()
        ml_filters = ml_filters[:, :, 0:15]
        return ml_filters

    def get_diff(self, array):
        num_reg = array.shape[0]
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            mat[i] = np.abs(array[i] - array[:])
        return mat

    def get_diff_hist(self, color):
        num_reg = len(self.rlist)
        hist = np.ones([num_reg, 256])
        for i in range(num_reg):
            hist[i][color[self.rlist[i]]] += 1
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            a = 2 * (hist[i] - hist[:])**2
            b = hist[i] + hist[:] + 1
            mat[i] = np.sum(a/b,axis = 1)
        return mat

    def dot(self, x, hist=False):
        if hist:
            diff = self.get_diff_hist(x)
        else:
            diff = self.get_diff(x)
        x = self.w * diff
        x = x * self.a[0]
        return x
