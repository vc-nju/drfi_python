import os
import cv2
import numpy as np
import pandas as pd


class Region2Csv():

    def __init__(self, features, rlist, img_type, img_id):
        self.features = features
        self.rlist = rlist
        self.img_type = img_type
        self.img_id = img_id
        self.neigh = features.utils.edge_neigh
        self.seg_ids = self.get_seg_ids()
        self.in_segs = self.get_in_segs()
        self.generate_csv()

    def get_seg_ids(self):
        """
        Find which seg_id could be used in this img_id. ( Because of the Discontinuous of seg_id. )
        """
        seg_ids = []
        for i in range(40):
            if os.path.exists("data/{}_coco2pic/{}_{}.png".format(self.img_type, self.img_id, i)):
                seg_ids.append(i)
        return seg_ids

    def get_in_segs(self):
        """
        Find whether two super region in the same segs or not.
        return:
            - in_segs: np( [num of segs, num of super_regions] ) 
                        if lower than 20% of region size in seg:
                            it will be -1, 
                        elif upper than 80% of region size in seg:
                            it will be 1,
                        else: 
                            it will be 0.
        """
        in_segs = np.zeros(
            [len(self.seg_ids), len(self.rlist)], dtype=np.uint8)
        for i in range(len(self.seg_ids)):
            seg_id = self.seg_ids[i]
            path = "data/{}_coco2pic/{}_{}.png".format(
                self.img_type, self.img_id, seg_id)
            seg = cv2.imread(path)[0]
            _seg = np.zeros_like(seg)
            _seg[seg == 127] = 1
            _seg[seg == 128] = 1
            for j in range(len(self.rlist)):
                r = self.rlist[i]
                in_size = np.sum(_seg[r[0], r[1]])
                if in_size < 0.2 * len(r[0]):
                    in_segs[i, j] = -1
                elif in_size > 0.8 * len(r[0]):
                    in_segs[i, j] = 1
        return in_segs

    def generate_csv(self):
        """
        Each line of CSV will be like this:
            | img_id | seg_id | region_id i | region_id j | is same_region | 222-dim features |
        """
        data = []
        line_data = np.zeros([1, 5 + 222])
        for seg_id in self.seg_ids:
            for region_i in range(len(self.rlist)):
                for region_j in self.neigh[region_i]:
                    i = self.seg_ids.index(seg_id)
                    if self.in_segs[i, region_i] == 0 or self.in_segs[i, region_j] == 0:
                        continue
                    is_same_region = (
                        self.in_segs[i, region_i] + self.in_segs[i, region_j])/2
                    line_data = np.zeros_like(line_data)
                    line_data[0, 0] = self.img_id
                    line_data[0, 1] = seg_id
                    line_data[0, 2] = region_i
                    line_data[0, 3] = region_j
                    line_data[0, 4] = is_same_region
                    line_data[0, 5: 5+35] = self.features.reg_features[region_i]
                    line_data[0, 40: 40 +
                              29] = self.features.con_features[region_i]
                    line_data[0, 69: 69 +
                              29] = self.features.bkp_features[region_i]
                    line_data[0, 98: 98 +
                              35] = self.features.reg_features[region_j]
                    line_data[0, 133: 133 +
                              29] = self.features.con_features[region_j]
                    line_data[0, 162: 162 +
                              29] = self.features.bkp_features[region_j]
                    line_data[0, 191: 191+29 +
                              7] = self.features.comb_features[region_i][region_j]
                    data.append(line_data)
        data = np.concatenate(data)
        df = pd.DataFrame(data)
        path = "data/csv/{}/{}.csv".format(self.img_type, self.img_id)
        df.to_csv(path, header=None)
