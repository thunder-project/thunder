from thunder import ThunderContext
from thunder import SourceExtraction

from test_utils import PySparkTestCase


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
