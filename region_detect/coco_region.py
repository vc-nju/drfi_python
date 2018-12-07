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
        try:
            assert(number not in number_list)
            imgIds = coco.getImgIds(imgIds=[number])
            img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
            I = io.imread(img['coco_url'])
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=0)
            anns = coco.loadAnns(annIds)
            anns[0]['segmentation']
            assert(len(anns) < 33)
        except:
            continue
        path = 'data/{}_origin/{}.png'.format(img_type, counter)
        cv2.imwrite(path, I)
        path = "data/{}_coco2pic/{}".format(img_type, counter)
        COCO_Utils.coco2pic(I, anns, path)
        counter += 1
        print("finished coco region detect {} {} imgIds:{}".format(img_type, counter, imgIds))
        number_list.append(number)

def generate_coco_data():
    get_anns("train", 450)
    get_anns("val", 50)
