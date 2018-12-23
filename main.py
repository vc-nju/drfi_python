import cv2
import os
for i in range(5000):
    file_name = "data/MSRA-B/({}).jpg".format(i)
    im = cv2.imread(file_name)
    print(im.shape)
    file_name = "data/MSRA-B/({}).png".format(i)