import numpy as np


class Edge():
    def __init__(self, a, b, weight):
        self.a = a
        self.b = b
        self.weight = weight


class Universe():
    def __init__(self, im_size):
        self.num = im_size
        self.im_size = im_size
        self.table = np.zeros([im_size, 2])
        self.table[:, 0] = np.zeros(im_size) # rank
        self.table[:, 1] = np.ones(im_size) # size
        self.p_table = np.arange(im_size, dtype=np.int32) # p

    def find(self, x):
        y = x
        while y != self.p_table[y]:
            y = self.p_table[y]
        self.p_table[x] = y
        return y

    def join(self, x, y):
        if self.table[x, 0] > self.table[y, 0]:
            self.p_table[y] = x
            self.table[x, 1] += self.table[y, 1] 
        else:
            self.p_table[x] = y
            self.table[y, 1] += self.table[x, 1] 
            if self.table[x, 0] == self.table[y, 0]:
                self.table[y, 0] += 1
        self.num -= 1

    def find_all(self):
        map_list = []
        counter = 0
        # for i in range(len(self.p_table)):
        #     y = self.find(i)
        #     if map_array[y] == -1:
        #         map_array[y] = counter
        #         counter += 1
        #     rmat[i] = map_array[y]
        # assert(counter==self.num)
        for i in range(len(self.p_table)):
            map_list.append(self.find(i))
        map_list = list(set(map_list))
        map_array = np.zeros(self.im_size, dtype=np.int32)
        map_array[map_list] = [i for i in range(len(map_list))]
        rmat = self.p_table
        for i in range(len(rmat)):
            rmat[i] = map_array[rmat[i]]
        return rmat

