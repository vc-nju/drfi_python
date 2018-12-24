import os
import cv2
import numpy as np
import pandas as pd


class Region2Csv():

    @staticmethod
    def get_in_segs(rlist, seg_path):
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
        seg = cv2.imread(seg_path)[:, :, 0]
        _seg = np.zeros_like(seg)
        _seg[seg == 255] = 1
        for i in range(len(rlist)):
            r = rlist[i]
            in_size = np.sum(_seg[r])
            if in_size < 0.2 * len(r[0]):
                in_segs[i] = -1
            elif in_size > 0.8 * len(r[0]):
                in_segs[i] = 1
        return in_segs

    @staticmethod
    def generate_similar_csv(rlist, comb_features, seg_path, csv_path):
        in_segs = Region2Csv.get_in_segs(rlist, seg_path)
        """
        Each line of CSV will be like this:
            | is same_region | 222-dim features |
        """
        data = []
        for i, comb_f in enumerate(comb_features):
            region_i_id = i
            for j,  region_j_id in enumerate(comb_f["j_ids"]):
                # at the edge
                if in_segs[region_i_id] == 0 or in_segs[region_j_id] == 0:
                    continue
                # both out the seg
                elif in_segs[region_i_id] == in_segs[region_j_id] == -1:
                    continue
                is_same_region = (
                    in_segs[region_i_id] + in_segs[region_j_id])/2
                line_data = np.zeros([1, 1 + 222])
                line_data[0, 0] = is_same_region
                line_data[0, 1:] = comb_f["features"][j]
                data.append(line_data)
        data = np.concatenate(data)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=0)

    @staticmethod
    def generate_seg_csv(rlist, features93, seg_path, csv_path):
        in_segs = Region2Csv.get_in_segs(rlist, seg_path)
        """
        Each line of CSV will be like this:
            | is same_region | 93-dim features |
        """
        data = []
        for i, features in enumerate(features93):
            # at the edge
            line_data = np.zeros([1, 1 + 93])
            if in_segs[i] == 0:
                continue
            is_seg = (in_segs[i] + 1)/2
            line_data[0, 0] = is_seg
            line_data[0, 1:] = features
            data.append(line_data)
        if len(data) == 0:
            print("got noting in {}".format(csv_path))
            return
        data = np.concatenate(data)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=0)

    @staticmethod
    def combine_csv(path_list, all_csv_path):
        data = [pd.read_csv(path).values for path in path_list if os.path.exists(path)]
        data = np.concatenate(data, axis=0)
        df = pd.DataFrame(data)
        df.to_csv(all_csv_path, index=0)
