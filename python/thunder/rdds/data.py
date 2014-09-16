

class Data(object):

    def __init__(self, rdd):
        self.rdd = rdd

    def first(self):
        return self.rdd.first()

    def collet(self):
        return self.rdd.collect()

    def count(self):
        return self.rdd.count()

    def mean(self):
        return self.rdd.values().mean()

    def sum(self):
        return self.rdd.values().sum()

    def variance(self):
        return self.rdd.values().variance()

    def stdev(self):
        return self.rdd.values().stdev()

    def stats(self):
        return self.rdd.values().stats()

    def cache(self):
        self.rdd.cache()
