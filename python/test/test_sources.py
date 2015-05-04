from numpy import meshgrid, ndarray, array_equal, array, sqrt
from test_utils import LocalTestCase

from thunder.extraction.source import Source, SourceModel


class TestSourceConstruction(LocalTestCase):

    def test_source(self):
        """
        (SourceConstruction) create
        """
        s = Source([[10, 10], [10, 20]])
        assert(isinstance(s.coordinates, ndarray))
        assert(array_equal(s.coordinates, array([[10, 10], [10, 20]])))

    def test_source_with_values(self):
        """
        (SourceConstruction) create with values
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(array_equal(s.coordinates, array([[10, 10], [10, 20]])))
        assert(array_equal(s.values, array([1.0, 2.0])))


class TestSourceProperties(LocalTestCase):

    def test_center(self):
        """
        (SourceProperties) center
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(array_equal(s.center, [10, 15]))

    def test_bbox(self):
        """
        (SourceProperties) bounding box
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(array_equal(s.bbox, [10, 10, 10, 20]))

    def test_area(self):
        """
        (SourceProperties) area
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(s.area == 2.0)

    def test_polygon(self):
        """
        (SourceProperties) polygon
        """
        x, y = meshgrid(range(0, 10), range(0, 10))
        coords = zip(x.flatten(), y.flatten())
        s = Source(coords)
        assert(array_equal(s.polygon, [[0, 0], [9, 0], [9, 9], [0, 9]]))

    def test_restore(self):
        """
        (SourceProperties) remove lazy attributes
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(array_equal(s.center, [10, 15]))

        assert("center" in s.__dict__.keys())
        s.restore()
        assert("center" not in s.__dict__.keys())

        assert(array_equal(s.center, [10, 15]))
        assert("center" in s.__dict__.keys())
        s.restore(skip="center")
        assert("center" in s.__dict__.keys())


class TestSourceMethods(LocalTestCase):

    def test_merge(self):
        """
        (SourceMethods) merge
        """
        s1 = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        s2 = Source([[10, 30], [10, 40]], values=[4.0, 5.0])
        s1.merge(s2)
        assert(array_equal(s1.coordinates, [[10, 10], [10, 20], [10, 30], [10, 40]]))
        assert(array_equal(s1.values, [1.0, 2.0, 4.0, 5.0]))

        s1 = Source([[10, 10], [10, 20]])
        s2 = Source([[10, 30], [10, 40]])
        s1.merge(s2)
        assert(array_equal(s1.coordinates, [[10, 10], [10, 20], [10, 30], [10, 40]]))


class TestSourceConversion(LocalTestCase):

    def test_to_list(self):
        """
        (SourceConversion) to list
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        getattr(s, "center")
        assert(isinstance(s.tolist().center, list))
        getattr(s, "bbox")
        assert(isinstance(s.tolist().bbox, list))

    def test_to_array(self):
        """
        (SourceConversion) to array
        """
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(isinstance(s.toarray().center, ndarray))
        assert(isinstance(s.tolist().toarray().center, ndarray))
        assert(isinstance(s.tolist().toarray().bbox, ndarray))


class TestSourceComparison(LocalTestCase):

    def test_distance_source(self):
        """
        (SourceComparison) distance to source
        """
        s1 = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        s2 = Source([[20, 20], [20, 30]], values=[1.0, 2.0])
        assert(s1.distance(s2) == sqrt(200))

    def test_distance_array(self):
        """
        (SourceComparison) distance to array
        """
        s1 = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(s1.distance([20, 25]) == sqrt(200))
        assert(s1.distance(array([20, 25])) == sqrt(200))


class TestSourceModelComparison(LocalTestCase):

    def test_match_sources(self):
        """
        (SourceModelComparison) matching sources
        """
        s1 = Source([[10, 10], [10, 20]])
        s2 = Source([[20, 20], [20, 30]])
        s3 = Source([[20, 20], [20, 30]])
        s4 = Source([[10, 10], [10, 20]])
        s5 = Source([[15, 15], [15, 20]])

        sm1 = SourceModel([s1, s2])
        sm2 = SourceModel([s3, s4, s5])

        assert(sm1.match(sm2) == [1, 0])
        assert(sm2.match(sm1) == [1, 0, 0])