import cv2
import numpy as np
im = np.zeros([400,400], dtype=np.int32)
cv2.imshow("test", im)
cv2.waitKey()