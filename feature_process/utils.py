class Utils()
    def __init__(self, img3u, rlist):
        self.img3u = img3u
        self.rlist = rlist
        self.labimg3f = self.get_labimg3f()
        self.hsvimg3f = self.get_hsvimg3f()
        self.mlFilters15d = self.lmfilkernal()
        self.coord = self.get_coord()
        self.rvar = self.get_varchannel()
        self.vartex = self.get_vartex()
        self.varlbp = self.get_varlbp()


    def get_labimg3f(self):  # get lab channel
        img3f = self.img3u/255.0
        labimg3f = np.zeros(img3f.shape)
        cv2.cvtColor(img3f, labimg3f, cv2.CV_RGB2Lab)
        return labimg3f

    def get_hsvimg3f(self):  # get hsv channel
        img3f = self.img3u/255.0
        hsvimg3f = np.zeros(img3f.shape)
        cv2.cvtColor(img3f, hsvimg3f, cv2.CV_RGB2HSV)
        return hsvimg3f

    def get_coord(self):
        num_reg = len(self.rlist)
        coord = np.zeros((num_reg, 7))
        for i in range(num_reg):
            # edit: remove for-loop
            sum_y_x = np.sum(np.array(self.rlist[i], dtype=np.int32))
            num_pix = len(self.rlist[i])
            coord[i][0:2] = sum_y_x//num_pix
            sortbyx = [_x for _x in sorted(self.rlist[i], key=lambda x: x[1])]
            sortbyy = [_y for _y in sorted(self.rlist[i], key=lambda x: x[0])]
            tenth = int(num_pix*0.1)
            ninetith = int(num_pix*0.9)
            coord[i][2:6] = [sortbyy[tenth], sortbyx[tenth], sortbyy[ninetith], sortbyx[ninetith]]
            ratio = float(sortbyy[-1] - sortbyy[0]) / float(sortbyx[-1] - sortbyx[0])
            coord[i][6] = ratio
        return coord

    def get_varchannel(self): #get average and variance value of rgb,lab and hsv in each region
        num_reg = len(self.rlist)
        raver = np.zeros((num_reg, 9))
        rvar = np.zeros((num_reg, 9))
        B, G, R = cv2.split(self.img3u)
        #imglab = RegionFeature.get_labimg3f(img3u)
        #imghsv = RegionFeature.get_hsvimg3f(img3u)
        L, a, b = cv2.split(self.imglab3f)
        H, S, V = cv2.split(self.imghsv3f)
        # imgchan = [R,G,B,L,a,b,H,S,V]
        imgchan = np.append([R, G, B, L, a, b, H, S, V], axis=2)
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            raver[i, :] = np.sum(imchan[self.rlist[i]], axis=1) / num_pix
            rvar[i, :] = np.sum((imgchan[self.rlist[i]] - raver[i,:])**2 ,axis=1)/num_pix
        return rvar

    def matread(self,file):
        info_name = file.read(5)
        headData = np.zeros(3, dtype=int32)
        for i in range(3):
            headData[i] = struct.unpack('i', file.read(4))
        total = headData[0]*headData[1]*headData[2]
        mat = np.zeros(total, dtype=np.int8)
        for i in range(total):
            mat[i] = file.read(1)
        mat.reshape((headData[0], headData[1], headData[2]))
        return mat

    def lmfilkernal(self,file="DrfiModel.data"):
        with open(file, 'rb') as f:
                file_name = f.read(9)
                _N = np.zeros(3) #_N,_NumN,_NumT,according to drfi_cpp realization
                for i in range(3):
                    number = struct.unpack('i', f.read(4))
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
        return _mlFilters15d #LM filters,the most important parameter of texture filter response

    def get_vartex(self): #the average value of texture filter response
        num_reg = len(self.rlist)
        avertex = np.zeros([num_reg, 15])
        vartex = np.zeros([num_reg, 15])
        mlFilters15d = self.mlFilters15d
        gray1u = np.zeros([self.img3u.shape[0], self.img3u.shape[1]], dtype=np.int8)
        gray1d = np.zeros([self.img3u.shape[0], self.img3u.shape[1]], dtype=np.float)
        cv2.cvtColor(self.img3u, gray1u, cv2.RGB2GRAY)
        gray1d = gray1u.astype(np.float) / 255.0
        imtext1d = np.zeros([gray1d.shape[0], gray1d.shape[1], 15])
        #mlFilters15d is the convolution kernal of LM filters mentioned in the paper
        cv2.filter2D(gray1d, imtext1d[:, :, i], cv2.CV_F64, mlFilters15d[:, :, i], (0, 0), 0.0, cv2.BORDER_REPLICATE)
        for i in range(num_reg):
            imtext1ds = imtext1d[self.rlist[i]]
            avertex[i] = np.sum(imtext1ds, axis=1)/len(self.rlist[i])
            vartex[i] = np.sum((imtext1ds - avertex)**2, axis=1)/len(self.rlist[i])
        return vartex

    def get_varlbp(self): #get the variance value of lbp feature in each region
        num_reg = len(self.rlist)
        varlbp = np.zeros(num_reg)
        averlbp = np.zeros(num_reg)
        gray1u = np.zeros((self.img3u.shape[0], self.img3u.shape[1]), np.int8)
        cv2.cvtColor(self.img3u, gray1u, cv2.RGB2GRAY)
        n_points = 8
        radius = 1
        METHOD = 'uniform'
        lbp = local_binary_pattern(self.img3u, n_points, radius,METHOD) #the lbp map of original picture
        # n_bins = int(lbp.max() + 1)
        # hist,_ = np.histogram(lbp,density=true,bins=n_bins,range=(0,n_bins))
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            for j in range(num_pix):
                x = self.rlist[i][j][0]
                y = self.rlist[i][j][1]
                averlbp[i] += lbp[x][y]
            averlbp[i] /= num_pix
            for j in range(num_pix):
                x = self.rlist[i][j][0]
                y = self.rlist[i][j][1]
                varlbp[i] += (lbp[x][y] - averlbp)**2
            varlbp[i] /= num_pix
        return varlbp