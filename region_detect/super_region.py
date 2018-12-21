import cv2
import numpy as np

from .utils import Edge, Universe

MIN_REGION_SIZE = 300


class Super_Region():
    @staticmethod
    def guass_filter(path):
        im = cv2.imread(path)
        kernel = np.ones((5, 5), np.float32)/25
        dst = cv2.filter2D(im, -1, kernel)
        dst = cv2.filter2D(dst, -1, kernel)
        return dst

    @staticmethod
    def get_edges(im):
        edges = []
        height = im.shape[0]
        width = im.shape[1]
        for y in range(height):
            for x in range(width):
                p1 = [y, x]
                p2_list = []
                if x < width - 1:
                    p2_list.append([y, x+1])
                if y < height - 1:
                    p2_list.append([y+1, x])
                if x < width - 1 and y < height - 1:
                    p2_list.append([y+1, x+1])
                if x < width - 1 and y > 0:
                    p2_list.append([y-1, x+1])
                if not p2_list:
                    pass
                for p2 in p2_list:
                    diff = np.sqrt(np.sum((im[y][x] - im[p2[0], p2[1]])**2))
                    edges.append(Edge(p1, p2, diff))
        edges.sort(key=lambda x: x.weight)
        return edges

    @staticmethod
    def get_region(path, c):
        """
        This method will return a List which contains all super_regions.
        args:
            - path: the img's path. like: "../data/77.jpg"
            - c: the thresholds: like: 166.
        return:
            -rlist = [
                        [ (y1, y2, y3,), (x1, x2, x3,) ],  # points in super_region0  
                        ... ,
                     ]
        """
        im = Super_Region.guass_filter(path)
        edges = Super_Region.get_edges(im)
        height = im.shape[0]
        width = im.shape[1]
        im_size = height * width
        u = Universe(np.ones(im_size), im_size)
        thresholds = np.ones(im_size) * c
        for e in edges:
            a = e.a[0] * width + e.a[1]
            b = e.b[0] * width + e.b[1]
            a = u.find(a)
            b = u.find(b)
            if a != b and e.weight <= thresholds[a] and e.weight <= thresholds[b]:
                u.join(a, b)
                a = u.find(a)
                thresholds[a] = e.weight + c / u.elts[a].size
        rlist = []
        index = 0
        # use index_array to map the p to index
        index_array = np.ones(im_size, dtype=np.int32)*-1
        for y in range(height):
            for x in range(width):
                p = u.find(y * width + x)
                if index_array[p] == -1:
                    index_array[p] = index
                    rlist.append([(), ()])
                    index += 1
                rlist[index_array[p]][0] += (y,)
                rlist[index_array[p]][1] += (x,)
        rmat = np.zeros(im.shape[0:2], dtype=np.int32)
        for i in range(len(rlist)):
            rlist[i] = tuple(rlist[i])
            rmat[rlist[i]] = i
        return rlist, rmat

    @staticmethod
    def combine_region(similarity, c, rlist, rmat):
        num_reg = len(rlist)
        elt_sizes = [len(r[0]) for r in rlist]
        u = Universe(elt_sizes, num_reg)
        thresholds = np.ones(num_reg) * c
        for i in range(num_reg):
            thresholds[i] /= u.elts[i].size
        edges = []
        print(similarity)
        for i in range(num_reg - 1):
            for j in range(i+1, num_reg):
                edges.append(Edge(i, j, similarity[i, j]))
        edges.sort(key=lambda x: x.weight)
        for e in edges:
            a = u.find(e.a)
            b = u.find(e.b)
            if a != b and e.weight <= thresholds[a] and e.weight <= thresholds[b]:
                u.join(a, b)
                a = u.find(a)
                thresholds[a] = e.weight + c / u.elts[a].size

        # # force minimum size of segmentation
        # for e in edges:
        #     a = u.find(e.a)
        #     b = u.find(e.b)
        #     if a != b and (u.elts[a].size < MIN_REGION_SIZE or u.elts[b].size < MIN_REGION_SIZE):
        #         u.join(a, b)

        # use index_array to map the p to index
        index_array = np.ones(num_reg, dtype=np.int32)*-1
        trans_array = np.zeros(num_reg)
        index = 0
        _rlist = []
        for i in range(num_reg):
            p = u.find(i)
            if index_array[p] == -1:
                index_array[p] = index
                _rlist.append([(), ()])
                index += 1
            _rlist[index_array[p]][0] += rlist[i][0]
            _rlist[index_array[p]][1] += rlist[i][1]
            trans_array[i] = index_array[p]
        _rmat = np.zeros_like(rmat)
        for i in range(rmat.shape[0]):
            for j in range(rmat.shape[1]):
                _rmat[i, j] = trans_array[rmat[i, j]]
        return _rlist, _rmat
