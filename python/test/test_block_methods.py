import unittest

from thunder import ThunderContext
from thunder import SourceExtraction

from test_utils import PySparkTestCase

_have_sima = False
try:
    import sima
except ImportError:
    pass
else:
    _have_sima = True

class TestBlockMethod(PySparkTestCase):

    def test_nmf(self):
        """
        (BlockMethod) nmf with defaults
        """
        tsc = ThunderContext(self.sc)
        data = tsc.makeExample('sources', dims=(60, 60), centers=[[20, 20], [40, 40]], noise=0.1, seed=42)

        model = SourceExtraction('nmf', componentsPerBlock=1).fit(data, size=(30, 30))

        # order is irrelevant, but one of these must be true
        ep = 0.50
        cond1 = (model[0].distance([20, 20]) < ep) and (model[1].distance([40, 40]) < ep)
        cond2 = (model[0].distance([40, 40]) < ep) and (model[1].distance([20, 20]) < ep)
        assert(cond1 or cond2)

    @unittest.skipIf(not _have_sima, "SIMA not installed or not functional")
    def test_sima(self):
        """
        (BlockMethod) with SIMA strategy
        """
        import sima.segment

        # construct the SIMA strategy
        simaStrategy = sima.segment.STICA(components=2)
        simaStrategy.append(sima.segment.SparseROIsFromMasks(min_size=20))
        simaStrategy.append(sima.segment.SmoothROIBoundaries())
        simaStrategy.append(sima.segment.MergeOverlapping(threshold=0.5))

        tsc = ThunderContext(self.sc)
        data = tsc.makeExample('sources', dims=(60, 60), centers=[[20, 15], [40, 45]], noise=0.1, seed=42)

        # create and fit the thunder extraction strategy
        strategy = SourceExtraction('sima', simaStrategy=simaStrategy)
        model = strategy.fit(data, size=(30, 30))

        # order is irrelevant, but one of these must be true
        ep = 1.5
        cond1 = (model[0].distance([20, 15]) < ep) and (model[1].distance([40, 45]) < ep)
        cond2 = (model[1].distance([20, 15]) < ep) and (model[0].distance([40, 45]) < ep)
        assert(cond1 or cond2)
