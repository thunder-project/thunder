from numpy import ndarray, asarray

from thunder.rdds.images import Images
from thunder.utils.common import checkparams


class Registration(object):
    """
    Class for constructing registration algorithms.

    Construct a registration algorthm by specifying its name,
    and passing any additional keyword arguments. The algorithm
    can then be used to fit registration parameters on an Images object.

    Parameters
    ----------
    method : string
        A registration method, options include 'crosscorr' and 'planarcrosscorr'
    """

    def __new__(cls, method, **kwargs):

        from thunder.imgprocessing.regmethods.crosscorr import CrossCorr, PlanarCrossCorr

        REGMETHODS = {
            'crosscorr': CrossCorr,
            'planarcrosscorr': PlanarCrossCorr
        }

        checkparams(method, REGMETHODS.keys())

        return REGMETHODS[method](kwargs)

    @staticmethod
    def load(file):
        """
        Load a registration model from a file specified using JSON.

        See also
        --------
        RegisterModel.save : specification for saving registration models
        RegisterModel.load : specifications for loading registration models
        """

        return RegistrationModel.load(file)


class RegistrationMethod(object):
    """
    Base class for registration methods
    """

    def __init__(self, *args, **kwargs):
        pass

    def prepare(self, **kwargs):
        raise NotImplementedError

    def isPrepared(self, images):
        raise NotImplementedError

    def getTransform(self, im):
        raise NotImplementedError

    def fit(self, images):
        """
        Compute registration parameters on a collection of images / volumes.

        Will return the estimated registration parameters to the driver in the form
        of a RegisterModel, which can then be used to transform Images data.

        Parameters
        ----------
        images : Images
            An Images object with the images / volumes to estimate registration for.

        Returns
        -------
        model : RegisterModel
            Registration params as a model that can be used to transform an Images object.

        See also
        --------
        RegisterModel : model for applying transformations
        """

        if len(images.dims.count) not in set([2, 3]):
                raise Exception('Number of image dimensions %s must be 2 or 3' % (len(images.dims.count)))

        self.isPrepared(images)

        # broadcast the registration model
        reg_bc = images.rdd.context.broadcast(self)

        # compute the transformations
        transformations = images.rdd.mapValues(lambda im: reg_bc.value.getTransform(im)).collectAsMap()

        # construct the model
        regmethod = self.__class__.__name__
        transclass = transformations.itervalues().next().__class__.__name__
        model = RegistrationModel(transformations, regmethod=regmethod, transclass=transclass)
        return model

    def run(self, images):
        """
        Compute and implement registration on a collection of images / volumes.

        This is a lazy operation that combines the estimation of registration
        with its implementaiton. It returns a new Images object with transformed
        images, and does not expose the registration parameters directly, see the
        'fit' method to obtain parameters directly.

        Parameters
        ----------
        images : Images
            An Images object with the images / volumes to apply registration to.

        Return
        ------
        Images object with registered images / volumes
        """

        if not (isinstance(images, Images)):
            raise Exception('Input data must be Images or a subclass')

        if len(images.dims.count) not in set([2, 3]):
            raise Exception('Number of image dimensions %s must be 2 or 3' % (len(images.dims.count)))

        self.isPrepared(images)

        # broadcast the reference
        reg_bc = images.rdd.context.broadcast(self)

        def fitandtransform(im, reg):
            t = reg.value.getTransform(im)
            return t.apply(im)

        newrdd = images.rdd.mapValues(lambda im: fitandtransform(im, reg_bc))

        return Images(newrdd).__finalize__(images)


class RegistrationModel(object):

    def __init__(self, transformations, regmethod=None, transclass=None):
        self.transformations = transformations
        self.regmethod = regmethod
        self.transclass = transclass

    def transform(self, images):
        """
        Apply the transformation to an Images object.

        Will apply the underlying dictionary of transformations to
        the images or volumes of the Images object. The dictionary acts as a lookup
        table specifying which transformation should be applied to which record of the
        Images object based on the key. Because transformations are small,
        we broadcast the transformations rather than using a join.

        See also
        --------
        Register : construct registration algorithms
        """

        from thunder.rdds.images import Images

        # broadcast the transformations
        transformations_bc = images.rdd.context.broadcast(self.transformations)

        # apply the transformations
        newrdd = images.rdd.map(lambda (k, im): (k, transformations_bc.value[k].apply(im)))
        return Images(newrdd).__finalize__(images)

    def save(self, file):
        """
        Serialize registration model to a file as text using JSON.

        Format is a dictionary, with keys 'regmethod' and 'transtype' specifying
        the registration method used and the transformation type (as strings),
        and 'transformations' containing the transformations. The exact format of the transformations
        will vary by type, but will always be a dictionary, with keys indexing into an Images object,
        and values containing the transformation parameters.

        Parameters
        ----------
        file : filename or file handle
            The file to write to

        """

        import json

        if hasattr(file, 'write'):
            f = file
        else:
            f = open(file, 'w')
        output = json.dumps(self, default=lambda v: v.__dict__)
        f.write(output)
        f.close()

    @staticmethod
    def load(file):
        """
        Deserialize registration model from a file containing JSON.

        Assumes a JSON formatted registration model, with keys 'regmethod' and 'transclass' specifying
        the registration method used and the transformation type as strings, and 'transformations'
        containing the transformations. The format of the transformations will depend on the type,
        but it should be a dictionary of key value pairs, where the keys are keys of the target
        Images object, and the values are arguments for reconstructing each transformation object.

        Parameters
        ----------
        file : str
            Name of a file to read from

        Returns
        -------
        model : RegisterModel
            Instance of a registration model
        """

        import json
        import importlib

        f = open(file, 'r')
        input = json.loads(f.read())

        # import the appropriate transformation class
        regmethod = str(input['regmethod'])
        classname = str(input['transclass'])
        transclass = getattr(importlib.import_module('thunder.imgprocessing.transformation'), classname)

        # instantiate the transformations and construct the model
        transformations = {int(k): transclass(**v) for k, v in input['transformations'].iteritems()}
        model = RegistrationModel(transformations, regmethod=regmethod, transclass=classname)
        return model

    def __repr__(self):
        out = "RegisterModel(method='%s', transtype='%s', transformations=%s)" % \
              (self.regmethod, self.transclass, self.transformations)
        return out[0:120] + " ..."





