import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


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
    plt.figure(0)
    plt.imshow(np.zeros_like(img, dtype=np.int8))
    plt.axis("off")
    c = 6.
    polygons = []
    colors = []
    segs = [ann[i]['segmentation'][0] for i in range(len(ann))]
    for seg in segs:
        poly = np.array(seg).reshape(int(len(seg)/2), 2)
        polygons.append(Polygon(poly))
        color = np.zeros((1, 3))
        color[0,0] += c/255. 
        color[0, 1] += (1.-c/255.)
        colors.append(color.tolist()[0])
        c += 6.
    ax = plt.gca()
    ax.set_autoscale_on(False)
    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=1)
    ax.add_collection(p)
    plt.savefig(path)
    plt.close(0)
    im = cv2.imread(path)
    a = im.shape + img.shape
    b = im.shape - img.shape
    im_ = np.zeros_like(img, dtype=np.int8)
    im_ = im[b[0]/2:a[0]/2, b[1]/2, a[1]/2, :]
    cv2.imwrite(path, im_)