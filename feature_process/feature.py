import numpy as np
import struct
import cv2
from .utils import Utils


class RegionFeature():
    def __init__(self,img3u,rlist,matrix):
        self._utils = Utils(img3u,rlist)
        self.regprop = self.get_regprop(matrix)

    def get_regprop(self): #property descriptor
        num_reg = len(self.rlist)
        regprop = np.zeros((num_reg,35),np.float32)
        rvarval = self._utils.rvar
        B,G,R = cv2.split(self.img3u)
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            sum_y = 0
            sum_x = 0
            edge_num = 0
            for j in range(num_pix):
                x = self.rlist[i][j][0]
                y = self.rlist[i][j][1]
                if self.matrix[x,y]!=self.matrix[x-1,y] or self.matrix[x,y]!=self.matrix[x+1,y] or self.matrix[x,y]!=self.matrix[x,y-1] or self.matrix[x,y]!=self.matrix[x,y+1]:
                    edge_num += 1
            coord = self._utils.coord
            for j in range(6):
                regprop[i][j] = coord[i][j]
            regprop[i][6] = edge_num #perimeter
            regprop[i][33] = num_pix #area
            regprop[i][7] = coord[i][6] #length-width ratio
            for k in range(9):
                regprop[i][k+8] = rvarval[i][k] #the variance of different channel
            vartex = self._utils.vartex
            for k in range(15):
                regprop[i][k+17] = vartex[i][k] #the variance of lm-filters
            varlbp = self._utils.varlbp
            regprop[i][32] = varlbp[i]
        neigharea = np.zeros(num_reg)
        sigmadist = 0.4
        for i in range(num_reg):
            x = regprop[i][0]
            y = regprop[i][1]
            for j in range(num_reg):
                _x = regprop[j][0]
                _y = regprop[j][1]
                neigharea[i] += math.exp(-((x - _x)**2 + (y - _y)**2)/sigmadist)
        return regprop

            
