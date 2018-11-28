import cv2
import math
#import struct
import numpy as np
# from skimage.feature import local_binary_pattern
from .lmfilter import make_lmfilter

class Utils():
    def __init__(self, rgb, rlist, rmat):
        self.height = rmat.shape[0] #for normalization
        self.width = rmat.shape[1]
        self.rgb = rgb
        self.rlist = rlist
        self.rmat = rmat
        self.lab = self.get_lab()
        self.hsv = self.get_hsv()
        self.coord = self.get_coord()
        self.color_var, self.color_avg = self.get_color_var()
        self.tex_var, self.tex_avg, self.tex = self.get_tex_var()
        # self.lbp_var, self.lbp = self.get_lbp_var()
        self.edge_nums = self.get_edge_nums()
        self.neigh_areas = self.get_neigh_areas()
        self.w = self.get_w()
        self.a = self.get_a() # normalization
        self.blist = self.get_background()

    def get_lab(self):
        lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab)
        return lab

    def get_hsv(self):
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        return hsv

    def get_coord(self):
        num_reg = len(self.rlist)
        coord = np.zeros([num_reg, 7])
        coord.astype('float')
        EPS = 1. # for division by zero in calculating the ratio
        for i in range(num_reg):
            sum_y_x = np.sum(np.array(self.rlist[i], dtype=np.int32),axis=1)
            num_pix = len(self.rlist[i][0])
            coord[i][0:2] = sum_y_x//num_pix
            coord[i][0] /= self.height #normalized mean position:y
            coord[i][1] /= self.width #normalized mean position:x
            sortbyy = sorted(self.rlist[i][0])
            sortbyx = sorted(self.rlist[i][1])
            tenth = int(num_pix*0.1)
            ninetith = int(num_pix*0.9)
            coord[i][2:6] = [sortbyy[tenth]/self.height, sortbyx[tenth]/self.width, #normalization
                sortbyy[ninetith]/self.height, sortbyx[ninetith]/self.width]
            ratio = float(sortbyy[-1] - sortbyy[0]) / \
                          float(sortbyx[-1] - sortbyx[0] + EPS)
            coord[i][6] = ratio
        return coord

    def get_color_var(self):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, 9])
        var = np.zeros([num_reg, 9])
        imgchan = np.concatenate([self.rgb, self.lab, self.hsv], axis=2)
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            avg[i, :] = np.sum(imgchan[self.rlist[i][0], self.rlist[i][1]]) / num_pix
            var[i, :] = np.sum((imgchan[self.rlist[i][0], self.rlist[i][1]] - avg[i, :])**2 ,axis=0)/num_pix
        return var, avg

    def get_tex_var(self):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, 15])
        var = np.zeros([num_reg, 15])
        ml_fiters = self.ml_kernal()
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float) / 255.0
        tex = np.zeros([gray.shape[0], gray.shape[1], 15])
        #for i in range(len(ml_fiters)):
        for i in range(15):
            #tex[:, :, i] = cv2.filter2D(gray, cv2.CV_64F, ml_fiters[:, :, i], (0, 0), 0.0, cv2.BORDER_REPLICATE)
            tex[:,:,i] = cv2.filter2D(gray,cv2.CV_64F,ml_fiters[:,:,i])
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            avg[i] = np.sum(tex[self.rlist[i][0], self.rlist[i][1]])/num_pix
            var[i] = np.sum((tex[self.rlist[i][0], self.rlist[i][1]] - avg[i])**2)/num_pix
        return var, avg, tex

    def get_lbp_var(self):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, 1])
        var = np.zeros([num_reg, 1])
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, 'uniform') 
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            avg[i] = np.sum(lbp[self.rlist[i]])/num_pix
            var[i] = np.sum((lbp[self.rlist[i]] - avg[i])**2)/num_pix
        return var, lbp

    def get_edge_nums(self):
        num_reg = len(self.rlist)
        edge_nums = np.zeros([num_reg, 1])
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            for j in range(num_pix):
                y = self.rlist[i][0][j]
                x = self.rlist[i][1][j]
                if x==0 or x==(self.width-1) or y==0 or y==(self.height-1):
                    edge_nums[i] += 1
                # to be fix
                else:
                    is_edge = self.rmat[y, x] != self.rmat[y-1, x] or self.rmat[y,x] != self.rmat[y+1, x] or self.rmat[y, x]!=self.rmat[y, x-1] or self.rmat[y,x]!=self.rmat[y,x+1]
                    if is_edge:
                        edge_nums[i] += 1
        return edge_nums/(self.width*self.height) #normalization
    
    def get_neigh_areas(self):
        num_reg = len(self.rlist)
        neigh_areas = np.zeros([num_reg,1])
        sigmadist = 0.4
        for i in range(num_reg):
            y = self.coord[i][0]
            x = self.coord[i][1]
            for j in range(num_reg):
                _y = self.coord[j][0]
                _x = self.coord[i][1]
                neigh_areas[i] += math.exp(-((x - _x)**2 + (y - _y)**2)/sigmadist)
        return neigh_areas/(self.width*self.height) #normalization

    def get_w(self):
        num_reg = len(self.rlist)
        pos = np.zeros([num_reg, 2])
        for i in range(num_reg):
            reg_array = np.array(self.rlist[i])
            pos[i, :] = np.sum(reg_array) / reg_array.shape[0]
        w = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                w[i, j] = np.exp(-np.sum((pos[i, :] - pos[j, :])**2)/2)
        return w

    def get_a(self):
        a = np.zeros([len(self.rlist), 1] )
        a[:, 0] = [float(len(r))/float(self.width*self.height) for r in self.rlist]
        return np.array(a)
        

    def get_background(self):
        blist = []
        y = [y_ for y_ in range(15)] + [y_ for y_ in range(self.rgb.shape[0]-15, self.rgb.shape[0])]
        x = [x_ for x_ in range(self.rgb.shape[1])]
        for y_ in y:
            blist += [ (y_, x_) for x_ in x ]
        y = [y_ for y_ in range(15, self.rgb.shape[0] - 15)]
        x = [x_ for x_ in range(15)] + [x_ for x_ in range(self.rgb.shape[1]-15, self.rgb.shape[1])]
        for y_ in y:
            blist += [ (y_, x_) for x_ in x ]
        return [blist]
    
    def ml_kernal(self):
        ml_filters = np.zeros([49,49,15])
        ml_filters = make_lmfilter()[:,:,0:15]
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
        # prevent div 0
        hist = np.ones([num_reg, 256])
        for i in range(num_reg):
            hist[i][color[self.rlist[i][0], self.rlist[i][1]]] += 1
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                a = 2 * (hist[i] - hist[j])**2
                b = hist[i] + hist[j]
                mat[i][j] = np.sum(a/b)
        return mat
    
    def dot(self, x, hist=False, bkg=False):
        if hist:
            diff = self.get_diff_hist(x)
        else:
            diff = self.get_diff(x)
        if bkg:
            x = self.a[-1].dot(self.w[-1])
            x = np.sum(x.dot(diff[-1]), axis=1)
        else:
            x = self.a.dot(self.w)
            x = np.sum(x.dot(diff), axis=1)
        return x

