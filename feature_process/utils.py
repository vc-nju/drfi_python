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
        self.lab =  cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        self.hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        self.tex = self.get_tex()
        self.lbp = self.get_lbp()
        self.coord = self.get_coord()
        imgchan = np.concatenate([self.rgb, self.lab, self.hsv], axis=2)
        self.color_avg, self.color_var = self.get_avg_var(imgchan)
        self.tex_avg, self.tex_var = self.get_avg_var(self.tex)
        self.lbp_avg, self.lbp_var = self.get_avg_var(self.lbp)
        self.edge_nums, self.edge_neigh, self.edge_point = self.get_edge_nums()
        self.edge_prop = self.get_edge_prop()
        self.neigh_areas = self.get_neigh_areas()
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
        tex = tex.astype(np.int8)
        return tex

    def get_lbp(self):
        num_reg = len(self.rlist)
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1.).astype(np.int32)
        return lbp

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
            coord[i][6] = ratio
        return coord

    def get_avg_var(self, a):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, a.shape[2]])
        var = np.zeros([num_reg, a.shape[2]])
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            for j in range(a.shape[2]):
                avg[i, j] = np.sum(a[:,:,j][self.rlist[i]]) / num_pix
                var[i, j] = np.sum((a[:,:,j][self.rlist[i]] - avg[i, j])**2)/num_pix
        return avg, var

    def get_edges(self):
        '''
        @description: 
        @param {None} 
        @return: edge_nums: the total edge of Ri, 
                 edge_neigh: the neighbor region of Ri, [(R1,R2,R3...), (R1,R2,R3...),]
                 edge_point: the point in edge between Ri and Rj, [[[(y1,y2,...yi,),(x1,x2,...xi,)],[(y1,y2,y3,...),(x1,x2,x3,...)]]]
                 the storage sequence is corrdinated to edge_neigh: for Ri, the nei_point[i][j] means the edge point between Ri and Rj
                 may be a bit confusing, good luck!
        '''
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
        edge_mat[edge_mat!=0] = 1
        generate_y_list = lambda y: [y+1, y-1, y, y, y+1, y+1, y-1, y-1]
        generate_x_list = lambda x: [x, x, x+1, x-1, x+1, x-1, x+1, x-1]
        shape = (rmat.shape[0], rmat.shape[1], 2, 8)
        y_x = np.zeros(shape, dtype=np.int32)
        for y in range(shape[0]):
            for x in range(shape[1]):
                y_x[y,x,0,:] = generate_y_list(y)
                y_x[y,x,1,:] = generate_x_list(x)
        edge_nums = []
        edge_neigh = []
        edge_point = []
        append_not_exist = lambda x, _list: _list.append(x) if x not in _list else _list
        for region in rlist:
            num = 0
            neighs = []
            points = []
            for y, x in zip(region[0], region[1]):
                for edge_direct in range(edge_mat.shape[2]):
                    if edge_mat[y, x, edge_direct] != 0:
                        y_ = y_x[y, x, 0, edge_direct]
                        x_ = y_x[y, x, 1, edge_direct]
                        num += 1
                        neigh_id = rmat[y_, x_]
                        if neigh_id not in neighs:
                            neighs.append(neigh_id)
                        p = (y_, x_,)
                        if p not in points:
                            points.append(p)
            edge_nums.append(num)
            assert(len(neighs) != 0)
            edge_neigh.append(neighs)
            _points = [(),()]
            for i in range(len(points)):
                _points[0] += (points[i][0],)
                _points[1] += (points[i][1],)
            edge_point.append(tuple(_points))
        # edge_nums /= self.width*self.height
        return edge_nums, edge_neigh, edge_point


    def get_edge_prop(self):
        """
        return:             
            edge_prob: the property of the edge in two neighbor regions
        """
        num_reg = len(self.rlist)
        edge_prop = np.zeros((num_reg, num_reg, 7))
        for i in range(num_reg):  # region i
            for k in range(len(self.edge_neigh[i])):
                j = self.edge_neigh[i][k]  # region j
                # the points in the edge between Ri and Rj
                edge_ij = self.edge_point[i][k]
                num_points = len(edge_ij[0])
                edge_prop[i, j, 0] = float(sum(edge_ij[0]) / num_points)
                edge_prop[i, j, 1] = float(sum(edge_ij[1]) / num_points)
                sortby_y = sorted(edge_ij[0])
                sortby_x = sorted(edge_ij[1])
                tenth = int(num_points * 0.1)
                ninetith = int(num_points * 0.9)
                edge_prop[i, j, 2] = sortby_y[tenth]
                edge_prop[i, j, 3] = sortby_x[tenth]
                edge_prop[i, j, 4] = sortby_y[ninetith]
                edge_prop[i, j, 5] = sortby_x[ninetith]
                edge_prop[i, j, 6] = float(
                    num_points / (self.width * self.height))
        return edge_prop

    def get_neigh_areas(self):
        num_reg = len(self.rlist)
        neigh_areas = np.zeros([num_reg, 1])
        sigmadist = 0.4
        for i in range(num_reg):
            for j in range(num_reg):
                diff = (self.coord[i][0:2] - self.coord[j][0:2])**2
                diff = np.sum(diff)
                neigh_areas[i] += np.exp(-1*diff/sigmadist)
        neigh_areas /= self.width*self.height
        return neigh_areas

    def get_w(self):
        num_reg = len(self.rlist)
        pos = np.zeros([num_reg, 2])
        for i in range(num_reg):
            reg_array = np.array(self.rlist[i])
            pos[i, :] = np.sum(reg_array, axis=1) / reg_array.shape[1]
        pos[:,0] /= self.height
        pos[:,1] /= self.width 
        w = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                diff = (pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2
                w[i, j] = np.exp(-1. * diff / 2)
        return w

    def get_a(self):
        a = np.zeros([len(self.rlist), 1])
        a[:, 0] = [float(len(r[0]))/float(self.width*self.height)
                   for r in self.rlist]
        return np.array(a)
    
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
        return [blist]

    def ml_kernal(self):
        ml_filters = makeLMfilters()
        ml_filters = ml_filters[:, :, 0:15]
        return ml_filters

    def get_diff(self, array):
        num_reg = array.shape[0]
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                mat[i][j] = np.abs(array[i] - array[j])
        return mat

    def get_diff_hist(self, color):
        num_reg = len(self.rlist)
        hist = np.ones([num_reg, 256])
        for i in range(num_reg):
            hist[i][color[self.rlist[i]]] += 1
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                a = 2 * (hist[i] - hist[j])**2
                b = hist[i] + hist[j] + 1.
                mat[i][j] = np.sum(a/b)
        return mat

    def dot(self, x, hist=False):
        if hist:
            diff = self.get_diff_hist(x)
        else:
            diff = self.get_diff(x)
        x = self.w * diff
        x = x * self.a[0]
        return x
