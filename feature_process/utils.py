class Utils()
   def __init__(self, rgb, rlist, rmat):
        self.rgb = rgb
        self.rlist = rlist
        self.rmat = rmat
        self.lab = self.get_lab()
        self.hsv = self.get_hsv()
        self.coord = self.get_coord()
        self.color_var, self.color_avg = self.get_color_var()
        self.tex_var, self.tex_avg, self.tex = self.get_tex_var()
        self.lbp_var, self.lbp = self.get_lbp_var()
        self.edge_nums = self.get_edge_nums()
        self.neigh_areas = self.get_neigh_areas()
        self.w = self.get_w()
        self.blist = self.get_background()
        self.a = [ len(r) for r in rlist]

    def get_lab(self):
        lab = cv2.cvtColor(self.rgb, cv2.CV_RGB2Lab)
        return lab

    def get_hsv(self):
        hsv = cv2.cvtColor(self.rgb, cv2.CV_RGB2HSV)
        return hsv

    def get_coord(self):
        num_reg = len(self.rlist)
        coord = np.zeros([num_reg, 7])
        for i in range(num_reg):
            sum_y_x = np.sum(np.array(self.rlist[i], dtype=np.int32))
            num_pix = len(self.rlist[i])
            coord[i][0:2] = sum_y_x//num_pix
            sortbyx = [_x for _x in sorted(self.rlist[i], key=lambda x: x[1])]
            sortbyy = [_y for _y in sorted(self.rlist[i], key=lambda x: x[0])]
            tenth = int(num_pix*0.1)
            ninetith = int(num_pix*0.9)
            coord[i][2:6] = [sortbyy[tenth], sortbyx[tenth],
                sortbyy[ninetith], sortbyx[ninetith]]
            ratio = float(sortbyy[-1] - sortbyy[0]) / \
                          float(sortbyx[-1] - sortbyx[0])
            coord[i][6] = ratio
        return coord

    def get_color_var(self):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, 9])
        var = np.zeros([num_reg, 9])
        imgchan = np.append([self.rgb, self.lab, self.hsv], axis=2)
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            avg[i, :] = np.sum(imchan[self.rlist[i]]) / num_pix
            var[i, :] = np.sum((imgchan[self.rlist[i]] - avg[i, :])**2 ,axis=0)/num_pix
        return var, avg

    def get_tex_var(self):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, 15])
        var = np.zeros([num_reg, 15])
        lm_fiters = self.lm_kernal()
        gray = cv2.cvtColor(self.rgb, cv2.RGB2GRAY)
        gray = gray.astype(np.float) / 255.0
        tex = np.zeros([gray.shape[0], gray.shape[1], 15])
        for i in range(len(lm_fiters)):
            tex[:, :, i] = cv2.filter2D(gray, cv2.CV_F64, lm_fiters[:, :, i], (0, 0), 0.0, cv2.BORDER_REPLICATE)
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            avg[i] = np.sum(tex[self.rlist[i]])/num_pix
            var[i] = np.sum((tex[self.rlist[i]] - avg)**2)/num_pix
        return var, avg, tex

    def get_lbp_var(self):
        num_reg = len(self.rlist)
        avg = np.zeros([num_reg, 1])
        var = np.zeros([num_reg, 1])
        lbp = local_binary_pattern(self.rgb, 8, 1, 'uniform') 
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            avg[i] = np.sum(lbp[self.rlist[i]])/num_pix
            var[i] = np.sum((lbp[self.rlist[i]] - avg[i])**2)/num_pix
        return lbp_var, lbp

    def get_edge_nums(self):
        num_reg = len(self.rlist)
        edge_nums = np.zeros([num_reg, 1])
        for i in range(num_reg):
            num_pix = len(self.rlist[i])
            for j in range(num_pix):
                y = self.rlist[i][j][0]
                x = self.rlist[i][j][1]
                # to be fix
                is_edge = rmat[x, y] != rmat[x-1, y] or rmat[x,y] != rmat[x+1, y] or rmat[x, y]!=rmat[x, y-1] or rmat[x,y]!=rmat[x,y+1]
                if is_edge:
                    edge_nums[i] += 1
        return edge_nums
    
    def get_neigh_areas(self):
        '''
        add your neigh_areas process here
        '''
        return neigh_areas

    def get_w(self):
        num_reg = len(self.rlist)
        pos = np.zeros([num_reg, 2])
        for i in range(num_reg):
            reg_array = np.array(self.rlist[i])
            pos[i, :] = np.sum(reg_array) / reg_array.shape[0]
        w = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                w[i, j] = np.exp(-np.sum((pos[i, :] - pos[j, :])**2)/2)
        return w

    def get_background(self):
        blist = []
        y = [y_ for y_ in range(15)] + [y_ for y_ in range(self.rgb.shpae[0]-15, self.rgb.shape[0])]
        x = [x_ for x_ in range(self.rgb.shape[1])]
        for y_ in y:
            blist += [ (y_, x_) for x_ in x ]
        y = [y_ for y_ in range(15: self.rgb.shape[0] - 15)]
        x = [x_ for x_ in range(15):] + [x_ for x_ in range(self.rgb.shape[1]-15, self.rgb.shape[1])]
        for y_ in y:
            blist += [ (y_, x_) for x_ in x ]
        return [blist]
        
    def mat_read(self, file):
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

    def lm_kernal(self, file="DrfiModel.data"):
        with open(file, 'rb') as f:
            f.read(9 + 4*3)
            [self.mat_read(f) for i in range(8)]
            lm_filters = self.mat_read(f)
        return lm_filters

    def get_diff(self, array):
        num_reg = array.shape[0]
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                mat[i][j] = np.abs(array[i] - array[j])
        return mat

    def get_diff_hist(self, color):
        num_reg = array.shape[0]
        # prevent div 0
        hist = np.ones([num_reg, 256])
        for i in range(num_reg):
            hist[i][color[self.rlist[i]]] += 1
        mat = np.zeros([num_reg, num_reg])
        for i in range(num_reg):
            for j in range(num_reg):
                a = 2 * (hist[i] - hist[j])**2
                b = hist[i] + hist[j]
                mat[i][j] = np.sum(a/b)
        return mat
    
    def dot(self, x, hist=False, bkg=False):
        if hist:
            diff = self.get_diff_hist(x)
        else:
            diff = self.get_diff(x)
        if bkg:
            x = self.a[-1].dot(self.w[-1])
            x = np.sum(x.dot(diff[-1]), axis=1)
        else:
            x = self.a.dot(self.w)
            x = np.sum(x.dot(diff), axis=1)
        return x