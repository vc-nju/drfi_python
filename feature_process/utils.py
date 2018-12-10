import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .lmfilter import make_lmfilter
import time
def tt(t):
    s = time.time()
    print(s-t)
    return s

class Utils():
    def __init__(self, rgb, rlist, rmat):
        self.height = rmat.shape[0]
        self.width = rmat.shape[1]
        self.num_reg = len(rlist)
        self.rlist = rlist
        self.rmat = rmat
        self.rgb = rgb
        t = time.time()
        self.lab = self.get_lab()
        t = tt(t)
        self.hsv = self.get_hsv()
        t = tt(t)
        self.coord = self.get_coord()
        t = tt(t)
        self.color_var, self.color_avg = self.get_color_var()
        t = tt(t)
        self.tex_var, self.tex_avg, self.tex = self.get_tex_var()
        t = tt(t)
        self.lbp_var, self.lbp = self.get_lbp_var()
        t = tt(t)
        # the number and the neighbor region of region[i]
        self.edge_nums, self.edge_neigh, self.edge_point = self.get_edge_nums()
        t = tt(t)
        self.edge_prop = self.get_edge_prop()
        t = tt(t)
        self.neigh_areas = self.get_neigh_areas()
        t = tt(t)
        self.w = self.get_w()
        t = tt(t)
        self.a = self.get_a()
        t = tt(t)
        self.blist = self.get_background()
        t = tt(t)
    def get_lab(self):
        lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab)
        return lab

    def get_hsv(self):
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        return hsv

    def get_coord(self):
        num_reg = self.num_reg
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

    def get_color_var(self):
        num_reg = self.num_reg
        avg = np.zeros([num_reg, 9])
        var = np.zeros([num_reg, 9])
        imgchan = np.concatenate([self.rgb, self.lab, self.hsv], axis=2)
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            for j in range(9):
                avg[i, j] = np.sum(
                    imgchan[:,:,j][self.rlist[i][0], self.rlist[i][1]]) / num_pix
                var[i, j] = np.sum(
                    (imgchan[:,:,j][self.rlist[i][0], self.rlist[i][1]] - avg[i, j])**2)/num_pix
        return var, avg

    def get_tex_var(self):
        num_reg = self.num_reg
        avg = np.zeros([num_reg, 15])
        var = np.zeros([num_reg, 15])
        ml_fiters = self.ml_kernal()
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float) / 255.0
        tex = np.zeros([gray.shape[0], gray.shape[1], 15])
        for i in range(15):
            tex[:, :, i] = cv2.filter2D(gray, cv2.CV_64F, ml_fiters[:, :, i])
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            for j in range(15):
                avg[i,j] = np.sum(tex[:,:,j][self.rlist[i][0], self.rlist[i][1]])/num_pix
                var[i,j] = np.sum(
                    (tex[:,:,j][self.rlist[i][0], self.rlist[i][1]] - avg[i,j])**2)/num_pix
        for i in range(15):
            tex_max = np.max(tex[:, :, i])
            tex_min = np.min(tex[:, :, i])
            tex[:, :, i] = (tex[:, :, i] - tex_min)/(tex_max - tex_min) * 255
        tex = tex.astype(np.int8)
        return var, avg, tex

    def get_lbp_var(self):
        num_reg = self.num_reg
        avg = np.zeros([num_reg, 1])
        var = np.zeros([num_reg, 1])
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1.)
        for i in range(num_reg):
            num_pix = len(self.rlist[i][0])
            avg[i] = np.sum(lbp[tuple(self.rlist[i])])/num_pix
            var[i] = np.sum((lbp[tuple(self.rlist[i])] - avg[i])**2)/num_pix
        return var, lbp.astype(np.int32)


    def get_edge_nums(self):
        """
        return:
            edge_nums: the total edge of Ri, 
            edge_neigh: the neighbor region of Ri, [(R1,R2,R3...), (R1,R2,R3...),]
            edge_point: the point in edge between Ri and Rj, [[[(y1,y2,...yi,),(x1,x2,...xi,)],[(y1,y2,y3,...),(x1,x2,x3,...)]]]
            the storage sequence is corrdinated to edge_neigh: for Ri, the nei_point[i][j] means the edge point between Ri and Rj
            may be a bit confusing, good luck!
        """
        num_reg = self.num_reg
        edge_nums = np.zeros([num_reg, 1])
        edge_neigh = []
        edge_point = []
        for i in range(num_reg):#for each region
            num_pix = len(self.rlist[i][0])
            # the neighbor region around region[i]
            edge_neigh.append(())
            # the edge point between Ri and Rj
            edge_point.append([])
            for j in range(num_pix):
                y = self.rlist[i][0][j]
                x = self.rlist[i][1][j]
                if x == 0 or x == (self.width-1) or y == 0 or y == (self.height-1):
                    edge_nums[i] += 1
                else:
                    is_edge = self.rmat[y-1, x] != i or self.rmat[y+1,x] != i or self.rmat[y, x-1] != i or self.rmat[y, x+1] != i
                    if is_edge:
                        if self.rmat[y-1, x] != i and (self.rmat[y-1, x] not in edge_neigh[i]):
                            _neigh = self.rmat[y-1, x]
                            # add the index of neighbor region
                            edge_neigh[i] += (self.rmat[y-1, x],)
                            edge_point[i].append([(),()])
                        elif (self.rmat[y+1, x],) != i and (self.rmat[y+1, x] not in edge_neigh[i]):
                            _neigh = self.rmat[y+1, x]
                            edge_neigh[i] +get_background= ((self.rmat[y+1, x]),)
                            edge_point[i].append([(),()])
                        elif self.rmat[y, x-1] != i and (self.rmat[y, x-1] not in edge_neigh[i]):
                            _neigh = self.rmat[y, x-1]
                            edge_neigh[i] += ((self.rmat[y, x-1]),)
                            edge_point[i].append([(),()])
                        elif self.rmat[y, x+1] != i and (self.rmat[y, x+1] not in edge_neigh[i]):
                            _neigh = self.rmat[y, x+1]
                            edge_neigh[i] += ((self.rmat[y, x+1]),)
                            edge_point[i].append([(),()])
                        edge_nums[i] += 1
                        neighbor = edge_neigh[i].index(_neigh)
                        edge_point[i][neighbor][0] += (y,)
                        edge_point[i][neighbor][1] += (x,)
        print(edge_neigh)
        edge_nums /= self.width*self.height
        return edge_nums, edge_neigh, edge_point


    def get_edge_prop(self):
        """
        return:             
            edge_prob: the property of the edge in two neighbor regions
        """
        num_reg = self.num_reg
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
        num_reg = self.num_reg
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
        num_reg = self.num_reg
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
        a = np.zeros([self.num_reg, 1])
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
        ml_filters = np.zeros([49, 49, 15])
        ml_filters = make_lmfilter()[:, :, 0:15]
        return ml_filters

    def get_diff(self, array):
        num_reg = array.shape[0]
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                mat[i][j] = np.abs(array[i] - array[j])
        return mat

    def get_diff_hist(self, color):
        num_reg = self.num_reg
        hist = np.ones([num_reg, 256])
        for i in range(num_reg):
            hist[i][color[self.rlist[i][0], self.rlist[i][1]]] += 1
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
