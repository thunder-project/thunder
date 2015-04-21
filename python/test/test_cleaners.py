from numpy import random
from test_utils import LocalTestCase

from thunder.extraction.source import Source, SourceModel
from thunder.extraction.cleaners import BasicCleaner


class TestBasicCleaner(LocalTestCase):

    def test_min(self):
        """
        (BasicCleaner) min area
        """
        list1 = [Source(random.randn(20, 2)) for _ in range(10)]
        list2 = [Source(random.randn(5, 2)) for _ in range(20)]
        sources = list1 + list2
        model = SourceModel(sources)

        c = BasicCleaner(minArea=10)
        newmodel = model.clean(c)

        assert(len(newmodel.sources) == 10)

    def test_max(self):
        """
        (BasicCleaner) max area
        """
        list1 = [Source(random.randn(20, 2)) for _ in range(10)]
        list2 = [Source(random.randn(5, 2)) for _ in range(20)]
        sources = list1 + list2
        model = SourceModel(sources)

        c = BasicCleaner(maxArea=10)
        newmodel = model.clean(c)

        assert(len(newmodel.sources) == 20)

    def test_min_max(self):
        """
        (BasicCleaner) min and max area
        """
        list1 = [Source(random.randn(20, 2)) for _ in range(10)]
        list2 = [Source(random.randn(10, 2)) for _ in range(20)]
        list3 = [Source(random.randn(15, 2)) for _ in range(5)]
        sources = list1 + list2 + list3
        model = SourceModel(sources)

        c = BasicCleaner(minArea=11, maxArea=19)
        newmodel = model.clean(c)

        assert(len(newmodel.sources) == 5)

    def test_min_max_chained(self):
        """
        (BasicCleaner) min and max area chained
        """
        list1 = [Source(random.randn(20, 2)) for _ in range(10)]
        list2 = [Source(random.randn(10, 2)) for _ in range(20)]
        list3 = [Source(random.randn(15, 2)) for _ in range(5)]
        sources = list1 + list2 + list3
        model = SourceModel(sources)

        c1 = BasicCleaner(minArea=11)
        c2 = BasicCleaner(maxArea=19)
        newmodel = model.clean([c1, c2])

        assert(len(newmodel.sources) == 5)


