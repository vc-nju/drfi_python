import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
'''
    load img and ann here
'''


class Edge():
    def __init__(self, a, b, weight):
        self.a = a
        self.b = b
        self.weight = weight


class Elt():
    def __init__(self, rank, p, size):
        self.rank = rank
        self.p = p
        self.size = size


class Universe():
    def __init__(self, im_size):
        self.num = im_size
        self.elts = [Elt(0, i, 1) for i in range(im_size)]

    def find(self, x):
        y = x
        while y != self.elts[y].p:
            y = self.elts[y].p
        self.elts[x].p = y
        return y

    def join(self, x, y):
        self.num -= 1
        _x, _y = x, y
        x, y = self.elts[x], self.elts[y]
        if x.rank > y.rank:
            y.p = _x
            x.size += y.size
        else:
            x.p = _y
            y.size += x.size
            if x.rank == y.rank:
                y.rank += 1


def coco2pic(img, ann, path):
    plt.figure()
    plt.imshow(np.zeros_like(img, dtype=np.int8))
    c = 1
    polygons = []
    colors = []
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        polygons.append(Polygon(poly))
        color = (np.ones((1, 3)) * c).tolist()[0]
        colors.append(color)
        c += 3
    ax = plt.gca()
    ax.set_autoscale_on(False)
    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=1)
    ax.add_collection(p)
    plt.savefig(path)
