import numpy as np
import struct
import cv2
import math
from skimage.feature import local_binary_pattern

class RegionFeature():
    @staticmethod
    def get_labimg3f(img3u): #get lab channel
        img3f = img3u/255.0
        labimg3f = np.zeros(img3f.shape)
        cv2.cvtColor(img3f,labimg3f,cv2.CV_RGB2Lab)
        return labimg3f
    
    @staticmethod
    def get_hsvimg3f(img3u): #get hsv channel
        img3f = img3u/255.0
        hsvimg3f = np.zeros(img3f.shape)
        cv2.cvtColor(img3f, hsvimg3f, cv2.CV_RGB2HSV)
        return hsvimg3f

    @staticmethod #@return value: numpy,the numberof regions by dimension of 9
    def get_averval(rlist,img3u): #get average value of rgb,lab and hsv in each region
        num_reg = len(rlist)
        raverval = np.zeros((num_reg,9))
        # update: use np instead of list
        imgchan = np.zeros([9, img3u.shape[0], img3u.shape[1]])
        imgchan[0:3] = cv2.split(img3u) # B,G,R
        imglab = RegionFeature.get_labimg3f(img3u)
        imghsv =RegionFeature.get_hsvimg3f(img3u)
        imgchan[3:6] = cv2.split(imglab) # L,a,b
        imgchan[6:] = cv2.split(imghsv) # H,S,V
        for i in range(num_reg):
            # fixed bug: change i+1 -> i, [j][2] -> [j][1], j[1] -> [j][0]
            num_pix = len(rlist[i])
            for j in range(num_pix):
                x = rlist[i][j][1]
                y = rlist[i][j][0]
                # fixed bug: [x,y] -> [y,x],  k+1 -> k
                raverval[i, :] += imgchan[:, y, x]
            raverval[i] /= num_pix
        return raverval

    @staticmethod
    def get_varval(rlist,img3u): #get variance value of rgb,lab and hsv in each region
        num_reg = len(rlist)
        B,G,R = cv2.split(img3u)
        imglab = RegionFeature.get_labimg3f(img3u)
        imghsv = RegionFeature.get_hsvimg3f(img3u)
        L,a,b = cv2.split(imglab)
        H,S,V = cv2.split(imghsv)
        raverval = RegionFeature.get_averval(rlist,img3u)
        rvarval = np.zeros((num_reg,9))
        imgchan = [R,G,B,L,a,b,H,S,V]
        for i in range(num_reg):
            num_pix = len(rlist[i+1])
            for j in range(num_pix):
                x = rlist[i+1][j][2]
                y = rlist[i+1][j][1]
                for k in range(9):
                    rvarval[i][k] += (imgchan[k+1][x,y] - raverval[i][k])**2
            rvarval[i] /= num_pix
        return rvarval

    @staticmethod
    def matread(file):
        info_name = file.read(5)
        headData = np.zeros(3)
        for i in range(3):
            headData[i] = struct.unpack('i',file.read(4))
        total = headData[0]*headData[1]*headData[2]
        mat = np.array(total,np.int8)
        for i in range(total):
            mat[i] = file.read(1)
        mat.reshape((headData[0],headData[1],headData[2]))
        return mat

    @staticmethod
    def lmfilkernal(file="DrfiModel.data"):
        with open(file,'rb') as f:
                file_name = f.read(9)
                _N = np.zeros(3) #_N,_NumN,_NumT
                for i in range(3):
                    number = struct.unpack('i',f.read(4))
                    _N[i] = number
                w = RegionFeature.matread(f)
                _segPara1d = RegionFeature.matread(f)
                _lDau1i = RegionFeature.matread(f)
                _rDau1i = RegionFeature.matread(f)
                _mBest1i = RegionFeature.matread(f)
                _nodeStatus1c = RegionFeature.matread(f)
                _upper1d = RegionFeature.matread(f)
                _avNode1d = RegionFeature.matread(f)
                _mlFilters15d = RegionFeature.matread(f)
                ndTree = RegionFeature.matread(f)
        return _mlFilters15d

    @staticmethod
    def get_vartex(rlist,img3u): #the average value of texture filter response
        num_reg = len(rlist)
        avertex = np.zeros((num_reg,15))
        vartex = np.zeros((num_reg,15))
        mlFilters15d = RegionFeature.lmfilkernal()
        gray1u = np.zeros((img3u.shape[0],img3u.shape[1]),np.int8)
        gray1d = np.zeros((img3u.shape[0],img3u.shape[1]),np.int8)
        cv2.cvtColor(img3u,gray1u,cv2.RGB2GRAY)
        gray1d = gray1u/255.0
        gray1d.astype(np.float64)
        imtext1d = np.zeros((gray1d.shape[0],gray1d.shape[1],15),np.float64)
        for i in range(15): #mlFilters15d is the convolution kernal of LM filters mentioned in the paper
            cv2.filter2D(gray1d,imtext1d[:,:,i],cv2.CV_F64,mlFilters15d[:,:,i],(0,0),0.0,cv2.BORDER_REPLICATE)
            for j in range(num_reg):
                for k in range(len(rlist[j])):
                    x = rlist[j][k][2]
                    y = rlist[j][k][1]
                    avertex[j][i] += imtext1d[x,y,i]
                avertex[j][i] /= len(rlist[j])
                for m in range(len(rlist[j])):
                    x = rlist[j][k][2]
                    y = rlist[j][k][1]
                    vartex[j][i] += (imtext1d[x,y,i] - avertex[j][i])**2
                vartex[j][i] /= len(rlist[j])
        return vartex 
            
    @staticmethod
    def get_varlbp(rlist,img3u): #get the variance value of lbp feature in each region
        num_reg = len(rlist)
        varlbp = np.zeros(num_reg)
        averlbp = np.zeros(num_reg)
        gray1u = np.zeros((img3u.shape[0],img3u.shape[1]),np.int8)
        cv2.cvtColor(img3u,gray1u,cv2.RGB2GRAY)
        n_points = 8
        radius = 1
        METHOD = 'uniform'
        lbp = local_binary_pattern(img3u,n_points,radius,METHOD) #the lbp map of original picture
        #n_bins = int(lbp.max() + 1)
        #hist,_ = np.histogram(lbp,density=true,bins=n_bins,range=(0,n_bins))
        for i in range(num_reg):
            num_pix = len(rlist[i])
            for j in range(num_pix):
                y = rlist[i+1][j+1][1]
                x = rlist[i+1][j+1][2]
                averlbp[i] += lbp[x][y]
            averlbp[i] /= num_pix
            for j in range(num_pix):
                y = rlist[i+1][j+1][1]
                x = rlist[i+1][j+1][2]
                varlbp[i] += (lbp[x][y] - averlbp)**2
            varlbp[i] /= num_pix
        return varlbp
   
    @staticmethod
    def get_regprop(rlist,matrix,im): #property descriptor
        num_reg = len(rlist)
        regprop = np.zeros((num_reg,35),np.float32)
        raverval = RegionFeature.get_averval(rlist,im)
        rvarval = RegionFeature.get_varval(rlist,im)
        B,G,R = cv2.split(im)
        for i in range(num_reg):
            num_pix = len(rlist[i])
            sum_y = 0
            sum_x = 0
            edge_num = 0
            for j in range(num_pix):
                x = rlist[i][j][2]
                y = rlist[i][j][1]
                sum_x += x
                sum_y += y
                if matrix[x,y]!=matrix[x-1,y] or matrix[x,y]!=matrix[x+1,y] or matrix[x,y]!=matrix[x,y-1] or matrix[x,y]!=matrix[x,y+1]:
                    edge_num += 1
            regprop[i][0] = sum_x / num_pix #average x
            regprop[i][1] = sum_y / num_pix #average y
            temp = sorted(rlist[i],key = lambda x:x[1])
            ten_x = (int)(num_pix*0.1)
            nin_x = (int)(num_pix*0.9)
            regprop[i][2] = temp[ten_x][2] #10th-point x 
            regprop[i][4] = temp[nin_x][2]
            _temp = sorted(rlist[i],key = lambda x:x[0])
            ten_y = (int)(num_pix*0.1)
            nin_y = (int)(num_pix*0.9)
            regprop[i][3] = temp[ten_x][1]
            regprop[i][5] = temp[nin_x][1]
            regprop[i][6] = edge_num #perimeter
            regprop[i][33] = num_pix #area
            ratio = (temp[num_pix][2] - temp[1][2]) / (_temp[num_pix][1] - _temp[1][1])
            regprop[i][7] = ratio #length-width ratio
            for k in range(9):
                regprop[i][k+8] = rvarval[i][k] #the variance of different channel
            vartex = RegionFeature.get_vartex(rlist,im)
            for k in range(15):
                regprop[i][k+17] = vartex[i][k] #the variance of lm-filters
            varlbp = RegionFeature.get_varlbp(rlist,im)
            regprop[i][32] = varlbp[i]
        neigharea = np.zeros(num_reg)
        sigmadist = 0.4
        for i in range(num_reg):
            x = regprop[i][0]
            y = regprop[i][1]
            for j in range(num_reg):
                _x = regprop[j][0]
                _y = regprop[i][1]
                neigharea[i] += math.exp(-((x - _x)**2 + (y - _y)**2)/sigmadist)
        return regprop

            