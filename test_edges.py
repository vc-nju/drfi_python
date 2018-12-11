import numpy as np
import pickle
def get_edges(rmat, rlist):
    '''
    @description: 
    @param {None} 
    @return: edge_nums: the total edge of Ri, 
             edge_neigh: the neighbor region of Ri, [(R1,R2,R3...), (R1,R2,R3...),]
             edge_point: the point in edge between Ri and Rj, [[[(y1,y2,...yi,),(x1,x2,...xi,)],[(y1,y2,y3,...),(x1,x2,x3,...)]]]
             the storage sequence is corrdinated to edge_neigh: for Ri, the nei_point[i][j] means the edge point between Ri and Rj
             may be a bit confusing, good luck!
    '''
    shape = (rmat.shape[0], rmat.shape[1], 8, )
    edge_mat = np.zeros(shape, dtype=np.int32)
    edge_mat[:-1, :, 0] += (rmat[1:, :] - rmat[:-1, :])
    edge_mat[1:, :, 1] += (rmat[:-1, :] - rmat[1:, :])
    edge_mat[:, :-1, 2] += (rmat[:, 1:] - rmat[:, :-1])
    edge_mat[:, 1:, 3] += (rmat[:, :-1] - rmat[:, 1:])
    edge_mat[:-1, :-1, 4] += (rmat[1:, 1:] - rmat[:-1, :-1])
    edge_mat[:-1, 1:, 5] += (rmat[1:, :-1] - rmat[:-1, 1:])
    edge_mat[1:, :-1, 6] += (rmat[:-1, 1:] - rmat[1:, :-1])
    edge_mat[1:, 1:, 7] += (rmat[:-1, :-1] - rmat[1:, 1:])
    edge_mat[edge_mat!=0] = 1
    generate_y_list = lambda y: [y+1, y-1, y, y, y+1, y+1, y-1, y-1]
    generate_x_list = lambda x: [x, x, x+1, x-1, x+1, x-1, x+1, x-1]
    shape = (rmat.shape[0], rmat.shape[1], 2, 8)
    y_x = np.zeros(shape, dtype=np.int32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            y_x[y,x,0,:] = generate_y_list(y)
            y_x[y,x,1,:] = generate_x_list(x)
    edge_nums = []
    edge_neigh = []
    edge_point = []
    append_not_exist = lambda x, _list: _list.append(x) if x not in _list else _list
    for region in rlist:
        num = 0
        neighs = []
        points = []
        for y, x in zip(region[0], region[1]):
            for edge_direct in range(edge_mat.shape[2]):
                if edge_mat[y, x, edge_direct] != 0:
                    y_ = y_x[y, x, 0, edge_direct]
                    x_ = y_x[y, x, 1, edge_direct]
                    num += 1
                    neigh_id = rmat[y_, x_]
                    if neigh_id not in neighs:
                        neighs.append(neigh_id)
                    p = (y_, x_,)
                    if p not in points:
                        points.append(p)
        edge_nums.append(num)
        assert(len(neighs) != 0)
        edge_neigh.append(neighs)
        _points = [(),()]
        for i in range(len(points)):
            _points[0] += (points[i][0],)
            _points[1] += (points[i][1],)
        edge_point.append(tuple(_points))
    # edge_nums /= self.width*self.height
    return edge_nums, edge_neigh, edge_point

if __name__ == "__main__":
    with open("rlist.rlist", "rb+") as file:
        rlist = pickle.load(file)
    with open("rmat.rmat", "rb+") as file:
        rmat = pickle.load(file)
    rmat = rmat.astype(np.int32)
    import time
    a = time.time()
    _, _ , _ = get_edges(rmat, rlist)
    print(time.time() - a)