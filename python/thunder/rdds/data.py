from numpy import int16

FORMATS = {
    'int16': int16,
    'float': float
}


class Data(object):

    def __init__(self, rdd):
        self.rdd = rdd

    def first(self):
        return self.rdd.first()

    def collet(self):
        return self.rdd.collect()

    def cache(self):
        self.rdd.cache()