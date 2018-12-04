import pylab
import skimage.io as io
import cv2
from pycocotools.coco import COCO

from .utils import coco2pic

def get_anns(img_type, img_num):
    """
    Used to trans coco data into regions. Save it to png.
    input:
        - img_type: "train", "val", "test"
    return:
        - None
    """
    pylab.rcParams["figure.figsize"] = (8.0, 10.0)
    dataDir = "coco"
    dataType = "{}2017".format(img_type)
    annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)
    coco=COCO(annFile)
    counter = 0
    while counter < img_num:
        number = np.random.randint(581782)
        imgIds = coco.getImgIds(imgIds = [number])
        try:
    	    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        except KeyError:
            continue
        I = io.imread(img["coco_url"])
        annIds = coco.getAnnIds(imgIds=img["id"], iscrowd=0)
        anns = coco.loadAnns(annIds)
        path = "{}_origin/{}.jpg".format(img_type, counter)
        cv2.imwrite(path, I)
        path = "{}_segmetation/{}.jpg".format(img_type, counter)
        coco2pic(I, anns, path)
        counter += 1

if __name__ == '__main__':
    get_anns("train", 350)
    get_anns("val", 50)
    get_anns("test", 100)
