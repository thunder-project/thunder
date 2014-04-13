from numpy import in1d, zeros, array, size
from scipy.io import loadmat
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation


class MassUnivariateClassifier(object):
    """Class for loading and classifying with classifiers"""

    def __init__(self, paramfile):
        """Initialize classifier using parameters derived from a Matlab file,
        or a python dictionary. At a minimum, must contain a "labels" field, with the
        label to classify at each time point. Can additionally include fields for
        "features" (which feature was present at each time point)
        and "samples" (which sample was present at each time point)

        :param paramfile: string of filename, or dictionary, containing parameters
        """
        if type(paramfile) is str:
            params = loadmat(paramfile, squeeze_me=True)
        elif type(paramfile) is dict:
            params = paramfile
        else:
            raise TypeError("Parameters for classification must be provided as string with file location, or dictionary")

        self.labels = params['labels']

        if 'features' in params:
            self.features = params['features']
            self.nfeatures = len(list(set(self.features.flatten())))
            self.samples = params['samples']
            self.sampleids = list(set(self.samples.flatten()))
            self.nsamples = len(self.sampleids)
        else:
            self.nfeatures = 1
            self.nsamples = len(self.labels)

    @staticmethod
    def load(paramfile, classifymode, cv=0):
        return CLASSIFIERS[classifymode](paramfile, cv)

    def get(self, x, set=None):
        pass

    def classify(self, data, set=None):

        if self.nfeatures == 1:
            perf = data.mapValues(lambda x: [self.get(x)])
        else:
            if set is None:
                set = [[self.features[0]]]
            if type(set) is int:
                set = [set]
            for i in set:
                assert array(in1d(self.features, i)).sum() != 0, "Feature set invalid"
            perf = data.mapValues(lambda x: map(lambda i: self.get(x, i), set))

        return perf


class GaussNaiveBayesClassifier(MassUnivariateClassifier):
    """Class for gaussian naive bayes classification"""

    def __init__(self, paramfile, cv):
        """Create classifier

        :param paramfile: string of filename or dictionary with parameters (see MassUnivariateClassifier)
        :param cv: number of cross validation folds (none if 0)
        :
        """
        MassUnivariateClassifier.__init__(self, paramfile)

        self.cv = cv
        self.func = GaussianNB()

    def get(self, x, set=None):
        """Compute classification performance"""

        y = self.labels
        if self.nfeatures == 1:
            X = zeros((self.nsamples, 1))
            X[:, 0] = x
        else:
            X = zeros((self.nsamples, size(set)))
            for i in range(0, self.nsamples):
                inds = (self.samples == self.sampleids[i]) & (in1d(self.features, set))
                X[i, :] = x[inds]

        if self.cv > 0:
            return cross_validation.cross_val_score(self.func, X, y, cv=self.cv).mean()
        else:
            ypred = self.func.fit(X, y).predict(X)
            return array(y == ypred).mean()

CLASSIFIERS = {
    'gaussnaivebayes': GaussNaiveBayesClassifier
}
