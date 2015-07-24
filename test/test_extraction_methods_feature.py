from thunder import ThunderContext
from thunder import SourceExtraction

from test_utils import PySparkTestCase


class TestFeatureMethod(PySparkTestCase):

    def test_local_max(self):
        """
        (FeatureMethod) localmax with defaults
        """
        tsc = ThunderContext(self.sc)
        data = tsc.makeExample('sources', dims=[60, 60], centers=[[10, 10], [40, 40]], noise=0.0, seed=42)
        model = SourceExtraction('localmax').fit(data)

        # order is irrelevant, but one of these must be true
        cond1 = (model[0].distance([10, 10]) == 0) and (model[1].distance([40, 40]) == 0)
        cond2 = (model[0].distance([40, 40]) == 0) and (model[1].distance([10, 10]) == 0)
        assert(cond1 or cond2)