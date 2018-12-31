import cv2
import numpy as np
import skimage
import scipy

# def de_gaussnoise(img_path):
#     img = cv2.imread(img_path)
#     dstimg = cv2.GaussianBlur(img,(5,5), 2)
#     return dstimg

def de_gaussnoise(img_path):
    img = cv2.imread(img_path)
    dstimg = scipy.signal.wiener(img, (5,5), 2)
    return dstimg

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(img, size):
    img = cv2.imread(img)
    img_output = np.zeros(img.shape)
    for i in range(3):
        img_mean = uniform_filter(img[:,:,i], (size, size))
        img_sqr_mean = uniform_filter(img[:,:,i]**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2
        overall_variance = variance(img[:,:,i])
        img_weights = img_variance / (img_variance + overall_variance)
        img_output[:,:,i] = img_mean + img_weights * (img[:,:,i] - img_mean)
    return img_output

def de_spnoise(img_path):
    img = cv2.imread(img_path)
    dstimg = cv2.medianBlur(img, 9)
    return dstimg

if __name__ == "__main__":
    sp_paths = ["./val_pic/sp_{}.jpg".format(i) for i in range(3001, 3021)]
    gauss_paths = ["./val_pic/gauss_{}.jpg".format(i) for i in range(3001, 3021)]
    speckle_paths = ["./val_pic/speckle_{}.jpg".format(i) for i in range(3001, 3021)]
    poisson_paths = ["./val_pic/poisson_{}.jpg".format(i) for i in range(3001, 3021)]

    denoise_sp_path = ["./denoise/de_spnoise_{}.jpg".format(i) for i in range(3001, 3021)]
    denoise_gauss_path = ["./denoise/de_gaussnoise_{}.jpg".format(i) for i in range(3001, 3021)]
    denoise_speckle_path = ["./denoise/de_specklenoise_{}.jpg".format(i) for i in range(3001, 3021)]
    denoise_poisson_path = ["./denoise/de_poissonnoise_{}.jpg".format(i) for i in range(3001, 3021)]

    for i in range(20):
        # cv2.imwrite(denoise_sp_path[i], de_spnoise(sp_paths[i]))
        # cv2.imwrite(denoise_gauss_path[i], de_spnoise(gauss_paths[i]))
        cv2.imwrite(denoise_speckle_path[i], lee_filter(speckle_paths[i], 5))

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 