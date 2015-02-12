""" Thunder JSON serialization utility """
import abc
import json


def _isNamedTuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return hasattr(obj, "_fields") and hasattr(obj, "_asdict") and callable(obj._asdict)


class Serializable(object):
    """Mixin class that provides JSON serialization services to classes inheriting from it

    Inheriting from Serializable makes it easy to store class instances in a human
    readable JSON format and then recover the original object instance. This abstract
    class provides serialize() and save() instance methods, along with deserialize() and
    load() class methods. Serialize() and deserialize() convert to and from a python
    dictionary representation that can then be easily processed by python's standard JSON
    module. Save() and load() persist and load objects to and from files on the local
    file system, wrapping calls to serialize() and deserialize().

    Note that this class is NOT intended to provide fully general pickling capabilities.
    Rather, it is designed to make it very easy to convert small classes containing model
    properties to a human and machine parsable format for later analysis or visualization.

    A key feature of the class is that it can "pickle" data types that are not normally
    supported by Python's stock JSON dump() and load() methods. Supported datatypes include:
    list, set, tuple, namedtuple, OrderedDict, datetime objects, numpy ndarrays, and dicts
    with non-string (but still data) keys. Serialization is performed recursively, and
    descends into the standard python container types (list, dict, tuple, set).

    The class provides special-case handling for lists and dictionaries with values that
    are themselves all Serializable objects of the same type (provided they have a regular
    dictionary rather than __slots__). The JSON output for such homogenous containers will
    list the type of the contained objects only once; in the general case, each instance
    of a nested serializable type will be stored along with its type.

    There are a number of limitations on data structures that are currently supported.
    Unicode strings, for instance, are not yet supported. Objects that have both __slots__
    and __dict__ attributes (as can happen from inheritance, such as an object with
    __slots__ inheriting from Serializable itself) will have only the __slots__ attributes
    serialized. Object graphs containing loops will lead to infinite recursion, and should
    not be used with this class.

    Some of this code was adapted from these fantastic blog posts by Chris
    Wagner and Sunil Arora:

      http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html
      http://sunilarora.org/serializable-decorator-for-python-class/

    Examples
    --------

      class Visitor(Serializable):
          def __init__(self, ipAddr = None, agent = None, referrer = None):
              self.ip = ipAddr
              self.ua = agent
              self.referrer= referrer
              self.time = datetime.datetime.now()

      origVisitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

      # Serialize the object
      pickledVisitor = origVisitor.serialize()  # returns dictionary

      # Restore object from dictionary
      recovVisitor = Visitor.deserialize(pickledVisitor)

    """
    __metaclass__ = abc.ABCMeta

    def __serializeRecursively(self, data, numpyStorage):
        from collections import OrderedDict
        from numpy import ndarray
        import datetime

        dataType = type(data)
        if dataType in frozenset([type(None), bool, int, long, float, str]):
            return data
        elif dataType == list:
            # awkward special case - check for lists of homogeneous serializable objects
            isHomogeneousSerializableList = False
            elementType = None
            if data and isinstance(data[0], Serializable) and not hasattr(data[0], "__slots__"):
                elementType = type(data[0])
                isHomogeneousSerializableList = True
                for val in data:
                    if (not isinstance(val, Serializable)) or (type(val) != elementType):
                        isHomogeneousSerializableList = False
                        break
            if isHomogeneousSerializableList:
                # TODO: this assumes that we have a __dict__ to serialize from, needs to be modified to
                # work with __slots__ also
                return {
                    "py/hmgList": {
                        "type": elementType.__name__,
                        "module": elementType.__module__,
                        "data": [self.__serializeRecursively(val.__dict__, numpyStorage) for val in data]
                    }
                }
            else:
                # plain old list
                return [self.__serializeRecursively(val, numpyStorage) for val in data]
        elif dataType == OrderedDict:
            return {
                "py/collections.OrderedDict": [
                    [self.__serializeRecursively(k, numpyStorage),
                     self.__serializeRecursively(v, numpyStorage)] for k, v in data.iteritems()
                ]
            }
        elif _isNamedTuple(data):
            return {
                "py/collections.namedtuple": {
                    "type": dataType.__name__,
                    "fields": list(data._fields),
                    "values": [self.__serializeRecursively(getattr(data, f), numpyStorage) for f in data._fields]
                }
            }
        elif dataType == dict:
            # another awkward special case - check for homogenous serializable value types
            valueType = None
            isHomogenousSerializableValueType = False
            for val in data.itervalues():
                if isinstance(val, Serializable):
                    if valueType is None:
                        valueType = type(val)
                        isHomogenousSerializableValueType = True
                    elif type(val) != valueType:
                        isHomogenousSerializableValueType = False
                        break
                else:
                    isHomogenousSerializableValueType = False
                    break
            if isHomogenousSerializableValueType:
                return {"py/hmgDict": {
                    "type": valueType.__name__,
                    "module": valueType.__module__,
                    "data": [(self.__serializeRecursively(k, numpyStorage),
                              self.__serializeRecursively(val.__dict__, numpyStorage)) for (k, val) in data.iteritems()]
                    }
                }
            elif all(type(k) == str for k in data):  # string keys can be represented natively in JSON
                # avoid dict comprehension for py2.6 compatibility
                retVal = {}
                for k, v in data.iteritems():
                    retVal[k] = self.__serializeRecursively(v, numpyStorage)
                return retVal
            else:
                return {"py/dict": [(self.__serializeRecursively(k, numpyStorage),
                                     self.__serializeRecursively(v, numpyStorage)) for k, v in data.iteritems()]}
        elif dataType == tuple:                          # Recurse into tuples
            return {"py/tuple": [self.__serializeRecursively(val, numpyStorage) for val in data]}
        elif dataType == set:                            # Recurse into sets
            return {"py/set": [self.__serializeRecursively(val, numpyStorage) for val in data]}
        elif dataType == datetime.datetime:
            return {"py/datetime.datetime": str(data)}
        elif dataType == complex:
            return {"py/complex": [data.real, data.imag]}
        elif dataType == ndarray:
            if numpyStorage == 'ascii' or (numpyStorage == 'auto' and data.size < 1000):
                return {"py/numpy.ndarray": {
                    "encoding": "ascii",
                    "shape": data.shape,
                    "values": data.tolist(),
                    "dtype":  str(data.dtype)}}
            else:
                from base64 import b64encode
                return {"py/numpy.ndarray": {
                    "encoding": "base64",
                    "shape": data.shape,
                    "values": b64encode(data),
                    "dtype":  str(data.dtype)}}
        elif isinstance(data, Serializable):
            # nested serializable object
            return {"py/Serializable": {
                "type": dataType.__name__,
                "module": dataType.__module__,
                "data": data.serialize()
            }}

        raise TypeError("Type %s not data-serializable" % dataType)

    def serialize(self, numpyStorage='auto'):
        """
        Serialize this object to a python dictionary that can easily be converted
        to/from JSON using Python's standard JSON library.

        Parameters
        ----------

          numpyStorage: {'auto', 'ascii', 'base64' }, optional, default 'auto'
            Use to select whether numpy arrays will be encoded in ASCII (as
            a list of lists) in Base64 (i.e. space efficient binary), or to
            select automatically (the default) depending on the size of the
            array. Currently the Base64 encoding is selected if the array
            has more than 1000 elements.

        Returns
        -------

          The object encoded as a python dictionary with "JSON-safe" datatypes that is ready to
          be converted to a string using Python's standard JSON library (or another library of
          your choice).

        """
        # Check for unsupported class.
        # a mix of slots and dicts can happen from multiple inheritance
        # at the moment, this appears to be "working" - with the restriction that if there
        # is both __slots__ and __dict__, only the __slots__ attributes will be serialized / deserialized.
        # if hasattr(self, "__slots__") and hasattr(self, "__dict__"):
        #    raise TypeError("Cannot serialize a class that has attributes in both __slots__ and __dict__")

        # If this object has slots, we need to convert the slots to a dict before serializing them.
        if hasattr(self, "__slots__"):
            retVal = {}
            for k in self.__slots__:
                v = getattr(self, k)
                retVal[self.__serializeRecursively(k, numpyStorage)] = \
                    self.__serializeRecursively(v, numpyStorage)
            return retVal

        # Otherwise, we handle the object as though it has a normal __dict__ containing its attributes.
        else:
            retVal = {}
            for k, v in self.__dict__.iteritems():
                retVal[self.__serializeRecursively(k, numpyStorage)] = \
                    self.__serializeRecursively(v, numpyStorage)
            return retVal

    @classmethod
    def deserialize(cls, serializedDict):
        """
        Restore the object that has been converted to a python dictionary using an @serializable
        class's serialize() method.

        Parameters
        ----------

            serializedDict: a python dictionary returned by serialize()

        Returns
        -------

            A reconstituted class instance
        """
        def restoreRecursively(dct):
            from numpy import frombuffer, dtype, array
            from base64 import decodestring

            # First, check to see if this is an encoded entry
            dataKey = None
            if type(dct) == dict:
                filteredKeys = filter(lambda x: x.startswith("py/"), dct.keys())

                # If there is just one key with a "py/" prefix, that is the dataKey!
                if len(filteredKeys) == 1:
                    dataKey = filteredKeys[0]

            # If no data key is found, may have a primitive, a list, or a dictionary.
            if dataKey is None:
                if type(dct) == dict:
                    return dict([(restoreRecursively(k_), restoreRecursively(v_)) for (k_, v_) in dct.iteritems()])
                elif type(dct) == list:
                    return [restoreRecursively(val) for val in dct]
                else:
                    return dct

            # Otherwise, decode it!
            if "py/dict" == dataKey:
                return dict([(restoreRecursively(k_), restoreRecursively(v_)) for (k_, v_) in dct["py/dict"]])
            elif "py/tuple" == dataKey:
                return tuple([restoreRecursively(val) for val in dct["py/tuple"]])
            elif "py/set" == dataKey:
                return set([restoreRecursively(val) for val in dct["py/set"]])
            elif "py/collections.namedtuple" == dataKey:
                from collections import namedtuple
                data = restoreRecursively(dct["py/collections.namedtuple"])
                return namedtuple(data["type"], data["fields"])(*data["values"])
            elif "py/collections.OrderedDict" == dataKey:
                from collections import OrderedDict
                return OrderedDict(restoreRecursively(dct["py/collections.OrderedDict"]))
            elif "py/datetime.datetime" == dataKey:
                from dateutil import parser
                return parser.parse(dct["py/datetime.datetime"])
            elif "py/complex" == dataKey:
                data = dct["py/complex"]
                return complex(float(data[0]), float(data[1]))
            elif "py/hmgList" == dataKey:
                from importlib import import_module
                data = dct["py/hmgList"]
                className = data["type"]
                moduleName = data["module"]
                clazz = getattr(import_module(moduleName), className)
                return [clazz.deserialize(val) for val in data["data"]]
            elif "py/hmgDict" == dataKey:
                from importlib import import_module
                data = dct["py/hmgDict"]
                className = data["type"]
                moduleName = data["module"]
                clazz = getattr(import_module(moduleName), className)
                return dict([(restoreRecursively(k_), clazz.deserialize(v_)) for (k_, v_) in data["data"]])
            elif "py/Serializable" == dataKey:
                from importlib import import_module
                data = dct["py/Serializable"]
                className = data["type"]
                moduleName = data["module"]
                clazz = getattr(import_module(moduleName), className)
                return clazz.deserialize(data["data"])
            elif "py/numpy.ndarray" == dataKey:
                data = dct["py/numpy.ndarray"]
                if data["encoding"] == "base64":
                    arr = frombuffer(decodestring(data["values"]), dtype(data["dtype"]))
                    return arr.reshape(data["shape"])
                elif data["encoding"] == "ascii":
                    data = dct["py/numpy.ndarray"]
                    return array(data["values"], dtype=data["dtype"])
                else:
                    raise TypeError("Unknown encoding key for numpy.ndarray: \"%s\"" % data["encoding"])

            # If no decoding scheme can be found, raise an exception
            raise TypeError("Could not de-serialize unknown type: \"%s\"" % dataKey)

        # First we must restore the object's dictionary entries.  These are decoded recursively
        # using the helper function above.
        restoredDict = {}
        for k in serializedDict.keys():
            restoredDict[k] = restoreRecursively(serializedDict[k])

        # Next we recreate the object. Calling the __new__() function here creates
        # an empty object without calling __init__(). We then take this empty
        # shell of an object, and set its dictionary to the reconstructed
        # dictionary we pulled from the JSON file.
        thawedObject = cls.__new__(cls)

        # If this class has slots, we must re-populate them one at a time
        if hasattr(cls, "__slots__"):
            for key in restoredDict.keys():
                setattr(thawedObject, key, restoredDict[key])

        # Otherwise simply update the objects dictionary en masse
        else:
            thawedObject.__dict__ = restoredDict

        # Return the re-constituted class
        return thawedObject

    def save(self, f, numpyStorage='auto'):
        """Serialize wrapped object to a JSON file.

        Parameters
        ----------
        f : string path to file or open writable file handle
            The file to write to. A passed handle will be left open for further writing.
        """
        def saveImpl(fp, numpyStorage_):
            json.dump(self.serialize(numpyStorage=numpyStorage_), fp)
        if isinstance(f, basestring):
            with open(f, 'w') as handle:
                saveImpl(handle, numpyStorage)
        else:
            # assume f is a file
            saveImpl(f, numpyStorage)

    @classmethod
    def load(cls, f):
        """Deserialize object from a JSON file.

        Assumes a JSON formatted registration model, with keys 'regmethod' and 'transclass' specifying
        the registration method used and the transformation type as strings, and 'transformations'
        containing the transformations. The format of the transformations will depend on the type,
        but it should be a dictionary of key value pairs, where the keys are keys of the target
        Images object, and the values are arguments for reconstructing each transformation object.

        Parameters
        ----------
        f : string path to file or file handle
            File to be read from

        Returns
        -------
        New instance of cls, deserialized from the passed file
        """
        def loadImpl(fp):
            dct = json.load(fp)
            return cls.deserialize(dct)

        if isinstance(f, basestring):
            with open(f, 'r') as handle:
                return loadImpl(handle)
        else:
            # assume f is a file object
            return loadImpl(f)
