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
from itertools import chain

try:
    from numpy import maximum, minimum, sqrt
except ImportError:
    maximum = max
    minimum = min
    sqrt = math.sqrt


class StatCounter(object):

    REQUIRED_FOR = {
        'mean': ('mu',),
        'sum': ('mu',),
        'min': ('minValue',),
        'max': ('maxValue',),
        'variance': ('mu', 'm2'),
        'sampleVariance': ('mu', 'm2'),
        'stdev': ('mu', 'm2'),
        'sampleStdev': ('mu', 'm2'),
        'all': ('mu', 'm2', 'minValue', 'maxValue')
    }

    def __init__(self, values=(), stats='all'):
        self.n = 0L    # Running count of our values
        self.mu = 0.0  # Running mean of our values
        self.m2 = 0.0  # Running variance numerator (sum of (x - mean)^2)
        self.maxValue = float("-inf")
        self.minValue = float("inf")

        if isinstance(stats, basestring):
            stats = [stats]
        self.requiredAttrs = frozenset(chain().from_iterable([StatCounter.REQUIRED_FOR[stat] for stat in stats]))

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
            self.maxValue = maximum(self.maxValue, value) if not self.maxValue is None else value
        if self.__requires('minValue'):
            self.minValue = minimum(self.minValue, value)

        return self

    # checks whether the passed attribute name is required to be updated in order to support the
    # statistics requested in self.requestedStats.
    def __requires(self, attrName):
        return attrName in self.requiredAttrs

    # Merge another StatCounter into this one, adding up the internal statistics.
    def mergeStats(self, other):
        if not isinstance(other, StatCounter):
            raise Exception("Can only merge Statcounters!")

        if other is self:  # reference equality holds
            self.merge(copy.deepcopy(other))  # Avoid overwriting fields in a weird order
        else:
            # accumulator should only be updated if it's valid in both statcounters:
            self.requiredAttrs = set(self.requiredAttrs).intersection(set(other.requiredAttrs))

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
        if not all(attr in self.requiredAttrs for attr in StatCounter.REQUIRED_FOR[statName]):
            raise ValueError("'%s' stat not available, must be requested at StatCounter instantiation" % statName)

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
        return ("(count: %s, mean: %s, stdev: %s, max: %s, min: %s, required: %s)" %
                (self.count(), self.mean(), self.stdev(), self.max(), self.min(), str(tuple(self.requiredAttrs))))
