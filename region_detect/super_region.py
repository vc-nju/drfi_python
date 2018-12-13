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
        u = Universe(im_size)
        thresholds = np.ones(im_size) * c
        for e in edges:
            a = e.a[0] * width + e.a[1]
            b = e.b[0] * width + e.b[1]
            a = u.find(a)
            b = u.find(b)
            if a != b and e.weight <= thresholds[a] and e.weight <= thresholds[b]:
                u.join(a, b)
                a = u.find(a)
                thresholds[a] = e.weight + c / u.table[a,1]
        rmat= u.find_all().reshape([height, width])
        rlist = [[(),()] for i in range(u.num)]
        for y in range(height):
            for x in range(width):
                index = rmat[y, x]
                rlist[index][0] += (y,)
                rlist[index][1] += (x,)
        for i in range(len(rlist)):
            rlist[i] = tuple(rlist[i])
        return rlist, rmat
