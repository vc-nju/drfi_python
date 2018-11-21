import cv2
import numpy as np
from matplotlib import pyplot as plt


class Super_Region():
    @staticmethod
    def guass_filter(path="77.jpg"):
        im = cv2.imread("77.jpg")
        kernel = np.ones((5, 5), np.float32)/25
        dst = cv2.filter2D(im, -1, kernel)
        dst = cv2.filter2D(dst, -1, kernel)
        return dst

    @staticmethod
    def get_edges(im):
        edges = np.ones(im.shape)[np.newaxis, :]*100000
        for i in range(3):
            edges = np.insert(edges, 0, values=edges[0], axis=0)
        edges[0, :, 1:] = im[:, :-1]  # left
        edges[1, 1:, :] = im[:-1, :]  # up
        edges[2, 1:, 1:] = im[:-1, :-1]  # left-up
        edges[3, 1:, :-1] = im[:-1, 1:]  # right-up
        for i in range(4):
            edges[i] = (edges[i]-im)**2
        edges = np.sqrt(np.sum(edges, axis=3))
        return edges

    @staticmethod
    def get_region(im, thresholds=0.1):
        edges = Super_Region.get_edges(im)
        min_edges = np.min(edges, axis=0)
        directions = np.argmin(edges, axis=0)
        region = np.zeros(im.shape[0:2], dtype=np.int32)
        max_num = 0
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                if min_edges[y][x] < thresholds:
                    nums = [region[y][x-1], region[y-1][x], region[y-1][x-1]]
                    if x != im.shape[1]-1:
                        nums.append(region[y-1][x+1])
                    region[y][x] = nums[directions[y][x]]
                else:
                    max_num += 1
                    region[y][x] = max_num
        return region

    @staticmethod
    def matrix2list(matrix):
        max_num = np.max(matrix)
        rlist = [[] for i in range(max_num)]
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                rlist[matrix[y][x]-1].append([y, x, ])
        return rlist

    @staticmethod
    def get_region_list(path="77.jpg", thresholds=0.1):
        im = Super_Region.guass_filter(path)
        region = Super_Region.get_region(im, thresholds)
        rlist = Super_Region.matrix2list(region)
        return rlist
