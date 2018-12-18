import os
import cv2
import numpy as np
import pandas as pd


class Region2Csv():

    def __init__(self, features, rlist, img_type, img_id, seg_path):
        in_segs = self.get_in_segs(rlist, seg_path)
        self.generate_csv(features, rlist, img_type, img_id, in_segs)


    def get_in_segs(self, rlist, seg_path):
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
        in_segs = np.zeros(len(rlist))
        seg = cv2.imread(seg_path)[:,:,0]
        _seg = np.zeros_like(seg)
        _seg[seg == 0] = 1
        for i in range(len(rlist)):
            r = rlist[i]
            in_size = np.sum(_seg[r])
            if in_size < 0.2 * len(r[0]):
                in_segs[i] = -1
            elif in_size > 0.8 * len(r[0]):
                in_segs[i] = 1
        return in_segs

    def generate_csv(self,features, rlist, img_type, img_id, in_segs):
        """
        Each line of CSV will be like this:
            | img_id | seg_id | region_id i | region_id j | is same_region | 222-dim features |
        """
        data = []
        line_data = np.zeros([1, 4 + 222])
        neigh = features.utils.edge_neigh
        for i in range(len(rlist)):
            region_i_id = i
            for j in range(len(neigh[i])):
                region_j_id = neigh[i][j]
                # at the edge
                if in_segs[region_i_id] == 0 or in_segs[region_j_id] == 0:
                    continue
                # both out the seg
                elif in_segs[region_i_id] == in_segs[region_j_id] == -1:
                    continue
                is_same_region = (in_segs[region_i_id] + in_segs[region_j_id])/2
                line_data = np.zeros_like(line_data)
                line_data[0, 0] = img_id
                line_data[0, 1] = region_i_id
                line_data[0, 2] = region_j_id
                line_data[0, 3] = is_same_region
                line_data[0, 4:4+35] = features.reg_features[region_i_id]
                line_data[0, 39:39+29] = features.con_features[region_i_id]
                line_data[0, 68:68+29] = features.bkp_features[region_i_id]
                line_data[0, 97:97+35] = features.reg_features[region_j_id]
                line_data[0, 132:132+29] = features.con_features[region_j_id]
                line_data[0, 161:161+29] = features.bkp_features[region_j_id]
                line_data[0, 190:190+29+7] = features.comb_features[region_i_id][j]
                data.append(line_data)
        if(len(data) == 0):
            print("Got noting in {} {}".format(img_type, img_id))
            return
        data = np.concatenate(data)
        df = pd.DataFrame(data)
        path = "data/csv/{}/{}.csv".format(img_type, img_id)
        df.to_csv(path, index=0)

def combine_csv():
    data = []
    for i in range(450):
        path = "data/csv/train/{}.csv".format(i)
        if os.path.exists(path):
            data.append(pd.read_csv(path).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("data/csv/train/all.csv")
    data = []
    for i in range(50):
        path = "data/csv/val/{}.csv".format(i)
        if os.path.exists(path):
            data.append(pd.read_csv(path).values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data)
    df.to_csv("data/csv/val/all.csv")