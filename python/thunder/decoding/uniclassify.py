"""
Class for mass-univariate classification
"""

from numpy import in1d, zeros, array, size, float64, asarray

from thunder.rdds.series import Series


class MassUnivariateClassifier(object):
    """
    Base class for performing classification.

    Assumes a Series with array data such that each entry belongs to one
    of several categories. Uses independent classifiers to train and predict.
    Example usage: determining how well each individual neural signal
    predicts a behavior.

    Parameters
    ----------
    paramfile : str or dict
        A Python dictionary, or MAT file, containing parameters for training classifiers.
        At minimum must contain a "labels" field, with the label to classify at each
        time point. Can additionally include fields for "features" (which feature
        was present at each time point) and "samples" (which sample was present
        at each time point)


    Attributes
    ----------
    `labels` : array
        Array containing the label for each time point

    `features` : array
        Which feature was present at each time point

    `nfeatures` : int, optional
        Number of features

    `samples` : array
        Which sample does each time point belong to

    `sampleIds` : list
        Unique samples

    `nsamples` : int
        Number of samples
    """

    def __init__(self, paramFile):
        from scipy.io import loadmat
        if type(paramFile) is str:
            params = loadmat(paramFile, squeeze_me=True)
        elif type(paramFile) is dict:
            params = paramFile
        else:
            raise TypeError("Parameters for classification must be provided as string with file location," +
                            " or dictionary")

        self.labels = params['labels']

        if 'features' in params:
            self.features = params['features']
            self.nfeatures = len(list(set(self.features.flatten())))
            self.samples = params['samples']
            self.sampleIds = list(set(self.samples.flatten()))
            self.nsamples = len(self.sampleIds)
        else:
            self.nfeatures = 1
            self.nsamples = len(self.labels)

    @staticmethod
    def load(paramFile, classifyMode, **opts):
        from thunder.utils.common import checkParams
        checkParams(classifyMode.lower(), CLASSIFIERS.keys())
        return CLASSIFIERS[classifyMode.lower()](paramFile, **opts)

    def get(self, x, featureSet=None):
        pass

    def fit(self, data, featureSet=None):
        """
        Run classification on each record in a data set

        Parameters
        ----------
        data: Series or a subclass (e.g. RowMatrix)
            Data to perform classification on, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        featureset : array, optional, default = None
            Which features to use

        Returns
        -------
        perf : Series
            The performance of the classifer for each record
        """

        if not isinstance(data, Series):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if self.nfeatures == 1:
            perf = data.rdd.mapValues(lambda x: [self.get(x)])
        else:
            if featureSet is None:
                featureSet = [[self.features[0]]]
            for i in featureSet:
                assert array([item in i for item in self.features]).sum() != 0, "Feature set invalid"
            perf = data.rdd.mapValues(lambda x: asarray(map(lambda j: self.get(x, j), featureSet)))

        return Series(perf, index=featureSet).__finalize__(data)


class GaussNaiveBayesClassifier(MassUnivariateClassifier):
    """Classifier for Gaussian Naive Bayes classification.

    Parameters
    ----------
    paramfile : str or dict
        Parameters for classification, see MassUnivariateClassifier

    cv : int
        Folds of cross-validation, none if 0
    """

    def __init__(self, paramfile, cv=0):
        MassUnivariateClassifier.__init__(self, paramfile)
        from sklearn.naive_bayes import GaussianNB
        from sklearn import cross_validation
        self.cvFunc = cross_validation.cross_val_score
        self.cv = cv
        self.func = GaussianNB()

    def get(self, x, featureSet=None):
        """
        Compute classification performance

        Parameters
        ----------
        x : array
            Data for a single record

        featureset : array, optional, default = None
            Which features to use

        Returns
        -------
        perf : scalar
            Performance of the classifier on this record
        """

        y = self.labels
        if self.nfeatures == 1:
            X = zeros((self.nsamples, 1))
            X[:, 0] = x
        else:
            X = zeros((self.nsamples, size(featureSet)))
            for i in range(0, self.nsamples):
                inds = (self.samples == self.sampleIds[i]) & (in1d(self.features, featureSet))
                X[i, :] = x[inds]

        if self.cv > 0:
            return self.cvFunc(self.func, X, y, cv=self.cv).mean()
        else:
            ypred = self.func.fit(X, y).predict(X)
            perf = array(y == ypred).mean()
            return perf


class TTestClassifier(MassUnivariateClassifier):
    """
    Classifier for TTest classification.

    Parameters
    ----------
    paramfile : str or dict
     Parameters for classification, see MassUnivariateClassifier
    """

    def __init__(self, paramfile):
        MassUnivariateClassifier.__init__(self, paramfile)
        from scipy.stats import ttest_ind
        self.func = ttest_ind
        unique = list(set(list(self.labels)))
        if len(unique) != 2:
            raise TypeError("Only two types of labels allowed for t-test classificaiton")
        if unique != set((0, 1)):
            self.labels = array(map(lambda i: 0 if i == unique[0] else 1, self.labels))

    def get(self, x, featureSet=None):
        """
        Compute classification performance as a t-statistic

        Parameters
        ----------
        x : array
            Data for a single record

        featureset : array, optional, default = None
            Which features to use

        Returns
        -------
        t : scalar
            t-statistic for this record
        """

        if (self.nfeatures > 1) & (size(featureSet) > 1):
            X = zeros((self.nsamples, size(featureSet)))
            for i in range(0, size(featureSet)):
                X[:, i] = x[self.features == featureSet[i]]
            t = float64(self.func(X[self.labels == 0, :], X[self.labels == 1, :])[0])

        else:
            if self.nfeatures > 1:
                x = x[self.features == featureSet]
            t = float64(self.func(x[self.labels == 0], x[self.labels == 1])[0])

        return t


# keys should be all lowercase, access should call .lower()
CLASSIFIERS = {
    'gaussnaivebayes': GaussNaiveBayesClassifier,
    'ttest': TTestClassifier
}
