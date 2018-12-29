# -*- coding: utf-8 -*-
 
from numpy import *
from scipy import * 
import numpy as np 
import cv2
import skimage

def SaltAndPepper(src, percentage, _amount):
    img = skimage.io.imread(src)
    SP_Noise = skimage.util.random_noise(img, mode="s&p", amount=_amount, seed=None, clip=True, salt_vs_pepper=percentage)
    return SP_Noise
 
def addGaussianNoise(imgName, _var):
    img = skimage.io.imread(imgName) 
    Gaussian_Noise = skimage.util.random_noise(img, mode="gaussian", var=_var, seed=None, clip=True)
    return Gaussian_Noise

def addSpeckleNoise(imgName):
    img = skimage.io.imread(imgName)
    Speckle_Noise = skimage.util.random_noise(img, mode="speckle", var=2, seed=None, clip=True)
    return Speckle_Noise

def addPoissonNoise(imgName):
    img = skimage.io.imread(imgName)
    Poisson_Noise = skimage.util.random_noise(img, mode="poisson", seed=None, clip=True)
    return Poisson_Noise

if __name__ == "__main__":
    src_imgs = ["./data/MSRA-B/{}.jpg".format(i) for i in range(3001, 3021)]
    sp_paths = ["./val_pic/sp_{}.jpg".format(i) for i in range(3001, 3021)]
    gauss_paths = ["./val_pic/gauss_{}.jpg".format(i) for i in range(3001, 3021)]
    speckle_paths = ["./val_pic/speckle_{}.jpg".format(i) for i in range(3001, 3021)]
    poisson_paths = ["./val_pic/poisson_{}.jpg".format(i) for i in range(3001, 3021)]
    for i in range(20):
        srcImage = src_imgs[i]  
        # SaltAndPepper_noiseImage = SaltAndPepper(srcImage,0.5, 1.0) #再添加10%的椒盐噪声
        # gauss_noiseImage = addGaussianNoise(srcImage, 0.5) 
        speckle_noiseImage = addSpeckleNoise(srcImage)
        # poisson_noiseImage = addPoissonNoise(srcImage)

        # sp_path = sp_paths[i]
        # gauss_path = gauss_paths[i]
        speckle_path = speckle_paths[i]
        # poisson_path = poisson_paths[i]

        # skimage.io.imsave(sp_path, SaltAndPepper_noiseImage)
        # skimage.io.imsave(gauss_path, gauss_noiseImage)
        skimage.io.imsave(speckle_path, speckle_noiseImage)
        # skimage.io.imsave(poisson_path, poisson_noiseImage)

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

