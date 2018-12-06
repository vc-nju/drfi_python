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


class COCO_Utils:

    @staticmethod
    def coco2pic(img, ann, path):
        img_shape = img.shape
        segs = [ann[i]['segmentation'][0] for i in range(len(ann))]
        counter = 0
        for seg in segs:
            poly = np.array(seg).reshape(int(len(seg)/2), 2)
            polygon = Polygon(poly)
            _path = "{}_{}.png".format(path, counter)
            COCO_Utils.polygon2pic(img_shape, polygon, _path)

    @staticmethod
    def polygon2pic(img_shape, polygon, path):
        plt.figure(0)
        plt.imshow(np.zeros(img_shape))
        plt.axis("off")
        ax = plt.gca()
        ax.set_autoscale_on(False)
        color = np.ones((1, 3)) * 0.5
        color = color.tolist()[0]
        p = PatchCollection([polygon], facecolor=[color],
                            linewidths=0, alpha=1)
        ax.add_collection(p)
        plt.savefig(path)
        plt.close(0)
        COCO_Utils.rm_white(img_shape, path)

    @staticmethod
    def rm_white(img_shape, path):
        im = cv2.imread(path)
        _im = np.zeros(img_shape)
        white = True
        j = 0
        while(white):
            for i in range(im.shape[1]):
                white = (im[j, i, 0] == 255) and (im[j, i, 1] == 255) and (im[j, i, 2] == 255)
                if not white:
                    _im[:, :, :] = im[j: img_shape[0] + j, i: img_shape[1] + i, :]
                    break
            j += 1
        cv2.imwrite(path, _im)
