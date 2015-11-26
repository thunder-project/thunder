from numpy import inf, subtract


class Dimensions(object):
    """ Class for estimating and storing dimensions of data based on the keys """

    def __init__(self, values=[], n=3):
        self.min = tuple(map(lambda i: inf, range(0, n)))
        self.max = tuple(map(lambda i: -inf, range(0, n)))

        for v in values:
            self.merge(v)

    def merge(self, value):
        self.min = tuple(map(min, self.min, value))
        self.max = tuple(map(max, self.max, value))
        return self

    def mergeDims(self, other):
        self.min = tuple(map(min, self.min, other.min))
        self.max = tuple(map(max, self.max, other.max))
        return self

    @property
    def count(self):
        return tuple(map(lambda x: x + 1, map(subtract, self.max, self.min)))

    @classmethod
    def fromTuple(cls, tup):
        """ Generates a Dimensions object from the passed tuple. """
        mx = [v-1 for v in tup]
        mn = [0] * len(tup)
        return cls(values=[mx, mn], n=len(tup))

    def __str__(self):
        return str(self.count)

    def __repr__(self):
        return str(self.count)

    def __len__(self):
        return len(self.min)

    def __iter__(self):
        return iter(self.count)

    def __getitem__(self, item):
        return self.count[item]
