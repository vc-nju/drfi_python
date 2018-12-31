import numpy as np
import cv2

def get_miou(gt, img):
    gt = cv2.imread(gt)
    img = cv2.imread(img)
    C1 = 0
    C2 = 0
    C1_C2 = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j][0] != 0:
                C1 += 1
            if img[i][j][0] != 0:
                C2 += 1
            if gt[i][j][0] and img[i][j][0]:
                C1_C2 += 1
    return C1_C2 / (C1 + C2 - C1_C2)

if __name__ == "__main__":
    gt = '21.png'
    img = 'test.png'
    _img = cv2.imread(img)
    for i in range(_img.shape[0]):
        for j in range(_img.shape[1]):
            if _img[i][j][0] < 128:
                _img[i][j][0] = 0
    cv2.imwrite('test.png', _img)
    cv2.imshow('img', _img)
    cv2.waitKey(0)
    a = get_miou(gt, img)
    print(a)
            

