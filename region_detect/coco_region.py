import pylab
import skimage.io as io
import cv2
import numpy as np
from pycocotools.coco import COCO

from .utils import COCO_Utils

def get_anns(img_type, img_num):
    """
    Used to trans coco data into regions. Save it to several regions png.
    input:
        - img_type: "train", "val", "test"
    return:
        - None
    """
    dataDir = 'coco'
    dataType = '{}2017'.format(img_type)
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    counter = 0
    number_list = []
    while counter < img_num:
        number = np.random.randint(110, 581929)
        if number in number_list:
            continue
        imgIds = coco.getImgIds(imgIds=[number])
        try:
            img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        except KeyError:
            continue
        try:
            I = io.imread(img['coco_url'])
        except:
            continue
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=0)
        anns = coco.loadAnns(annIds)
        if len(anns) > 32:
            continue
        try:
            anns[0]['segmentation']
        except IndexError:
            continue
        path = '{}_origin/{}.png'.format(img_type, counter)
        cv2.imwrite(path, I)
        path = "{}_coco2pic/{}".format(img_type, counter)
        COCO_Utils.coco2pic(I, anns, path)
        counter += 1
        print("finished coco region detect {} {} imgIds:{}".format(img_type, counter, imgIds))
        number_list.append(number)

if __name__ == '__main__':
    get_anns("train", 350)
    get_anns("val", 50)
    get_anns("test", 100)
