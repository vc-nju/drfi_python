import numpy as np


def w(self):
    num_reg = len(self.rlist)
    pos = np.zeros([num_reg, 2])
    for i in range(num_reg):
        reg_array = np.array(self.rlist[i])
        pos[i, :] = np.sum(reg_array, axis=1) / reg_array.shape[0]
    w = np.zeros([num_reg, num_reg])
    for i in range(num_reg):
        for j in range(num_reg):
            w[i, j] = np.exp(-np.sum((pos[i, :] - pos[j, :])**2)/2)
    return w

def 
