#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file is ported from spark/util/StatCounter.scala
# Modified from pyspark's statcounter.py.

import copy
import math

try:
    from numpy import maximum, minimum, sqrt
except ImportError:
    maximum = max
    minimum = min
    sqrt = math.sqrt


class StatCounter(object):
    DEFAULT_STATS = frozenset(('mean', 'sum', 'min', 'max', 'variance', 'sampleVariance', 'stdev', 'sampleStdev'))

    STATS_REQUIRING = {
        'mu': ('mean', 'sum', 'variance', 'std', 'sampleVariance', 'sampleStdev'),
        'm2': ('variance', 'std', 'sampleVariance', 'sampleStdev'),
        'max': ('maxValue',),
        'min': ('minValue',)
    }

    def __init__(self, values=(), stats=DEFAULT_STATS):
        self.n = 0L    # Running count of our values
        self.mu = 0.0  # Running mean of our values
        self.m2 = 0.0  # Running variance numerator (sum of (x - mean)^2)
        self.maxValue = float("-inf")
        self.minValue = float("inf")
        self.requestedStats = stats

        for v in values:
            self.merge(v)

    # Add a value into this StatCounter, updating the internal statistics.
    def merge(self, value):
        self.n += 1
        if self.__requires('mu'):
            delta = value - self.mu
            self.mu += delta / self.n
            if self.__requires('m2'):
                self.m2 += delta * (value - self.mu)
        if self.__requires('maxValue'):
            self.maxValue = maximum(self.maxValue, value)
        if self.__requires('minValue'):
            self.minValue = minimum(self.minValue, value)

        return self

    def __requires(self, accumName):
        return any([stat in StatCounter.STATS_REQUIRING[accumName] for stat in self.requestedStats])

    # Merge another StatCounter into this one, adding up the internal statistics.
    def mergeStats(self, other):
        if not isinstance(other, StatCounter):
            raise Exception("Can only merge Statcounters!")

        if other is self:  # reference equality holds
            self.merge(copy.deepcopy(other))  # Avoid overwriting fields in a weird order
        else:
            # accumulator should only be updated if it's valid in both statcounters:
            self.requestedStats = set(self.requestedStats).intersection(set(other.requestedStats))

            if self.n == 0:
                self.n = other.n
                for attrname in ('mu', 'm2', 'maxValue', 'minValue'):
                    if self.__requires(attrname):
                        setattr(self, attrname, getattr(other, attrname))

            elif other.n != 0:
                if self.__requires('mu'):
                    delta = other.mu - self.mu
                    if other.n * 10 < self.n:
                        self.mu = self.mu + (delta * other.n) / (self.n + other.n)
                    elif self.n * 10 < other.n:
                        self.mu = other.mu - (delta * self.n) / (self.n + other.n)
                    else:
                        self.mu = (self.mu * self.n + other.mu * other.n) / (self.n + other.n)

                    if self.__requires('m2'):
                        self.m2 += other.m2 + (delta * delta * self.n * other.n) / (self.n + other.n)

                if self.__requires('maxValue'):
                    self.maxValue = maximum(self.maxValue, other.maxValue)
                if self.__requires('minValue'):
                    self.minValue = minimum(self.minValue, other.minValue)

                self.n += other.n
        return self

    # Clone this StatCounter
    def copy(self):
        return copy.deepcopy(self)

    def count(self):
        return self.n

    def __checkAvail(self, statName):
        if not statName in self.requestedStats:
            raise ValueError("'%s' stat not available, only: %s" % str(tuple(self.requestedStats)))

    def mean(self):
        self.__checkAvail('mean')
        return self.mu

    def sum(self):
        self.__checkAvail('sum')
        return self.n * self.mu

    def min(self):
        self.__checkAvail('min')
        return self.minValue

    def max(self):
        self.__checkAvail('max')
        return self.maxValue

    # Return the variance of the values.
    def variance(self):
        self.__checkAvail('variance')
        if self.n == 0:
            return float('nan')
        else:
            return self.m2 / self.n

    #
    # Return the sample variance, which corrects for bias in estimating the variance by dividing
    # by N-1 instead of N.
    #
    def sampleVariance(self):
        self.__checkAvail('sampleVariance')
        if self.n <= 1:
            return float('nan')
        else:
            return self.m2 / (self.n - 1)

    # Return the standard deviation of the values.
    def stdev(self):
        self.__checkAvail('stdev')
        return sqrt(self.variance())

    #
    # Return the sample standard deviation of the values, which corrects for bias in estimating the
    # variance by dividing by N-1 instead of N.
    #
    def sampleStdev(self):
        self.__checkAvail('sampleStdev')
        return sqrt(self.sampleVariance())

    def __repr__(self):
        return ("(count: %s, mean: %s, stdev: %s, max: %s, min: %s, requested: %s)" %
                (self.count(), self.mean(), self.stdev(), self.max(), self.min(), str(tuple(self.requestedStats))))
