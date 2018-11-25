class Edge():
    def __init__(self, a, b, weight):
        self.a = a
        self.b = b
        self.weight = weight


class Elt():
    def __init__(self, rank, p, size):
        self.rank = rank
        self.p = p
        self.size = size


class Universe():
    def __init__(self, im_size):
        self.num = im_size
        self.elts = [Elt(0, i, 1) for i in range(im_size)]

    def find(self, x):
        y = x
        while y != self.elts[y].p:
            y = self.elts[y].p
        self.elts[x].p = y
        return y

    def join(self, x, y):
        self.num -= 1
        _x, _y = x, y
        x, y = self.elts[x], self.elts[y]
        if x.rank > y.rank:
            y.p = _x
            x.size += y.size
        else:
            x.p = _y
            y.size += x.size
            if x.rank == y.rank:
                y.rank += 1
