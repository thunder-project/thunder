import shutil
import tempfile
from numpy import array
from numpy.testing import assert_array_almost_equal
from thunder.classification.util import MassUnivariateClassifier
from test_utils import PySparkTestCase


class ClassificationTestCase(PySparkTestCase):
    def setUp(self):
        super(ClassificationTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ClassificationTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestMassUnivariateClassification(ClassificationTestCase):
    """Test accuracy of mass univariate classification on small
    test data sets with either 1 or 2 features
    """

    def test_mass_univariate_classification_gnb_1d(self):
        """Simple classification problem, features
        perfectly predict labels in one dimension
        """
        X1 = array([-1, -1, -1.2, 1, 1, 1.2])
        X2 = array([-1, -1, 1.2, 1, 1, 1.2])
        labels = array([1, 1, 1, 2, 2, 2])
        params = dict([('labels', labels)])

        clf = MassUnivariateClassifier.load(params, "gaussnaivebayes", cv=0)

        data = self.sc.parallelize([X1])
        result = clf.classify(data).collect()
        assert_array_almost_equal(result[0], [1.0])

        data = self.sc.parallelize([X2])
        result = clf.classify(data).collect()
        assert_array_almost_equal(result[0], [5.0/6.0])

    def test_mass_univariate_classification_gnb_2d(self):
        """Classification problem in the plane with two features,
        first feature perfectly predicts labels but second doesn't,
        both features combined predict perfectly
        """

        X = array([-1, 1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2])
        features = array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        samples = array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        labels = array([1, 1, 1, 2, 2, 2])
        params = dict([('labels', labels), ('features', features), ('samples', samples)])
        clf = MassUnivariateClassifier.load(params, "gaussnaivebayes", cv=0)

        data = self.sc.parallelize([X])

        # first feature predicts perfectly
        result = clf.classify(data, 1).collect()
        assert_array_almost_equal(result[0], [1.0])

        # second feature gets one wrong
        result = clf.classify(data, 2).collect()
        assert_array_almost_equal(result[0], [5.0/6.0])

        # two features together predict perfectly
        result = clf.classify(data, [[1, 2]]).collect()
        assert_array_almost_equal(result[0], [1.0])

        # test iteration over multiple feature sets
        result = clf.classify(data, [[1, 2], [2]]).collect()
        assert_array_almost_equal(result[0], [1.0, 5.0/6.0])



