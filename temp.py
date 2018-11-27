import numpy as np
img = [ [1,2,3],
        [4,5,6],
        [7,8,9] ]
img = np.array(img)
rlist = [ [(1,2,), (0,1), (0,0)] ]
# print(img[rlist[0]])
d = ()
print(d + (1,) + (2,))