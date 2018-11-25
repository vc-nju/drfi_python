import cv2
import numpy as np
from .utils import Edge, Universe

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
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                p1 = [y, x]
                p2_list = []
                if x < im.shape[1] - 1:
                    p2_list.append([y, x+1])
                if y < im.shape[0] - 1:
                    p2_list.append([y+1, x])
                if x < im.shape[1] - 1 and y < im.shape[0] - 1:
                    p2_list.append([y+1, x+1])
                if x < im.shape[1] - 1 and y > 0:
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
        im = Super_Region.guass_filter(path)
        edges = Super_Region.get_edges(im)
        im_size = im.shape[0]*im.shape[1]
        u = Universe(im_size)
        thresholds = np.ones(im_size) * c
        for e in edges:
            a = e.a[0] * im.shape[0] + e.a[1]
            b = e.b[0] * im.shape[0] + e.b[1]
            a = u.find(a)
            b = u.find(b)
            if a != b and e.weight <= thresholds[a] and e.weight <= thresholds[b]:
                u.join(a, b)
                a = u.find(a)
                thresholds[a] = e.weight + c / u.elts[a].size
        rlist = []
        index = 0
        index_array = np.ones(im_size, dtype=np.int32)*-1
        region = np.zeros(im.shape[0:2], dtype=np.int32)
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                p = u.find(y * im.shape[0] + x)
                if index_array[p] == -1:
                    index_array[p] = index
                    rlist.append([])
                    index += 1
                rlist[index_array[p]].append([y, x, ])
                region[y, x] = index_array[p]
        # if u need show im
        # region = region.astype('float')
        # region = region / np.max(region)
        # cv2.imshow("Image", region)
        # cv2.waitKey(0)
        return rlist

