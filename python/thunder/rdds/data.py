

class Data(object):

    def __init__(self, rdd):
        self.rdd = rdd

    def first(self):
        return self.rdd.first()

    def collet(self):
        return self.rdd.collect()

    def cache(self):
        self.rdd.cache()