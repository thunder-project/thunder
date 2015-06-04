from numpy import meshgrid, ndarray, array_equal, array, sqrt, zeros, asarray, where, ones
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

    def test_source_fromMask_binary(self):
        """
        (SourceConstruction) from mask
        """
        mask = zeros((10, 10))
        mask[5, 5] = 1
        mask[5, 6] = 1
        mask[5, 7] = 1
        s = Source.fromMask(mask)
        assert(isinstance(s, Source))
        assert(isinstance(s.coordinates, ndarray))
        assert(array_equal(s.coordinates, array([[5, 5], [5, 6], [5, 7]])))
        assert(array_equal(s.mask((10, 10), binary=True), mask))
        assert(array_equal(s.mask((10, 10), binary=False), mask))

    def test_source_fromMask_values(self):
        """
        (SourceConstruction) from mask with values
        """
        mask = zeros((10, 10))
        mask[5, 5] = 0.5
        mask[5, 6] = 0.6
        mask[5, 7] = 0.7
        s = Source.fromMask(mask)
        assert(isinstance(s, Source))
        assert(isinstance(s.coordinates, ndarray))
        assert(isinstance(s.values, ndarray))
        assert(array_equal(s.coordinates, array([[5, 5], [5, 6], [5, 7]])))
        assert(array_equal(s.values, array([0.5, 0.6, 0.7])))
        assert(array_equal(s.mask((10, 10), binary=False), mask))

    def test_source_fromCoordinates(self):
        """
        (SourceConstruction) from coordinates
        """
        s = Source.fromCoordinates([[10, 10], [10, 20]])
        assert(isinstance(s.coordinates, ndarray))
        assert(array_equal(s.coordinates, array([[10, 10], [10, 20]])))


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

    def test_inbounds(self):
        """
        (SourceMethods) in bounds
        """
        # two dimensional
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        assert(s.inbounds([0, 0], [20, 20]) == 1)
        assert(s.inbounds([0, 0], [10, 10]) == 0.5)
        assert(s.inbounds([15, 15], [20, 20]) == 0)

        # three dimensional
        s = Source([[10, 10, 10], [10, 20, 20]], values=[1.0, 2.0])
        assert(s.inbounds([0, 0, 0], [20, 20, 20]) == 1)
        assert(s.inbounds([0, 0, 0], [10, 10, 20]) == 0.5)
        assert(s.inbounds([15, 15, 15], [20, 20, 20]) == 0)

    def test_crop(self):
        """
        (SourceMethods) crop
        """
        # without values
        s = Source([[10, 10], [10, 20]])
        assert(array_equal(s.crop([0, 0], [21, 21]).coordinates, s.coordinates))
        assert(array_equal(s.crop([0, 0], [11, 11]).coordinates, [[10, 10]]))
        assert(array_equal(s.crop([0, 0], [5, 5]).coordinates, []))

        # with values (two dimensional)
        s = Source([[10, 10], [10, 20]])
        assert(array_equal(s.crop([0, 0], [21, 21]).coordinates, s.coordinates))
        assert(array_equal(s.crop([0, 0], [11, 11]).coordinates, [[10, 10]]))
        assert(array_equal(s.crop([0, 0], [5, 5]).coordinates, []))

    def test_exclude(self):
        """
        (SourceMethods) exclude
        """
        # without values
        s = Source([[10, 10], [10, 20]])
        o = Source([[10, 20]])
        assert(array_equal(s.exclude(o).coordinates, [[10, 10]]))

        # with values (two dimensional)
        s = Source([[10, 10], [10, 20]], values=[1.0, 2.0])
        o = Source([[10, 20]])
        assert(array_equal(s.exclude(o).coordinates, [[10, 10]]))
        assert(array_equal(s.exclude(o).values, [1]))

        # with values (three dimensional)
        s = Source([[10, 10, 10], [10, 20, 20]], values=[1.0, 2.0])
        o = Source([[10, 20, 20]])
        assert(array_equal(s.exclude(o).coordinates, [[10, 10, 10]]))
        assert(array_equal(s.exclude(o).values, [1.0]))

    def test_dilate(self):
        """
        (SourceMethods) dilate
        """
        # make base source
        m = zeros((10, 10))
        m[5, 5] = 1
        m[5, 6] = 1
        m[6, 5] = 1
        m[4, 5] = 1
        m[5, 4] = 1
        coords = asarray(where(m)).T
        s = Source(coords)

        # dilating by 0 doesn't change anything
        assert(array_equal(s.dilate(0).coordinates, s.coordinates))
        assert(array_equal(s.dilate(0).bbox, [4, 4, 6, 6]))

        # dilating by 1 expands region but doesn't affect center
        assert(array_equal(s.dilate(1).center, s.center))
        assert(array_equal(s.dilate(1).area, 21))
        assert(array_equal(s.dilate(1).bbox, [3, 3, 7, 7]))
        assert(array_equal(s.dilate(1).mask().shape, [5, 5]))

        # manually construct expected shape of dilated source mask
        truth = ones((5, 5))
        truth[0, 0] = 0
        truth[4, 4] = 0
        truth[0, 4] = 0
        truth[4, 0] = 0
        assert(array_equal(s.dilate(1).mask(), truth))

    def test_outline(self):
        """
        (SourceMethods) outline
        """
        # make base source
        m = zeros((10, 10))
        m[5, 5] = 1
        m[5, 6] = 1
        m[6, 5] = 1
        m[4, 5] = 1
        m[5, 4] = 1
        coords = asarray(where(m)).T
        s = Source(coords)

        # compare outlines to manual results
        o1 = s.outline(0, 1).mask((10, 10))
        o2 = s.dilate(1).mask((10, 10)) - s.mask((10, 10))
        assert(array_equal(o1, o2))

        o1 = s.outline(1, 2).mask((10, 10))
        o2 = s.dilate(2).mask((10, 10)) - s.dilate(1).mask((10, 10))
        assert(array_equal(o1, o2))


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