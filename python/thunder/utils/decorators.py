""" Useful decorators that are used throughout the library """

def _isNamedTuple(obj):
  """Heuristic check if an object is a namedtuple."""
  return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def serializable(cls):
    '''The @serializable decorator can decorate any class to make it easy to store
    that class in a human readable JSON format and then recall it and recover
    the original object instance. Classes instances that are wrapped in this
    decorator gain the serialize() method, and the class also gains a
    deserialize() static method that can automatically "pickle" and "unpickle" a
    wide variety of objects like so:

      @serializable
      class Visitor():
          def __init__(self, ipAddr = None, agent = None, referrer = None):
              self.ip = ipAddr
              self.ua = agent
              self.referrer= referrer
              self.time = datetime.datetime.now()

      origVisitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

      #serialize the object
      pickledVisitor = origVisitor.serialize()

      #restore object
      recovVisitor = Visitor.deserialize(pickledVisitor)

    Note that this decorator is NOT designed to provide generalized pickling
    capabilities. Rather, it is designed to make it very easy to convert small
    classes containing model properties to a human and machine parsable format
    for later analysis or visualization. A few classes under consideration for
    such decorating include the Transformation class for image alignment and the
    Source classes for source extraction.

    A key feature of the @serializable decorator is that it can "pickle" data
    types that are not normally supported by Python's stock JSON dump() and
    load() methods. Supported datatypes include: list, set, tuple, namedtuple,
    OrderedDict, datetime objects, numpy ndarrays, and dicts with non-string
    (but still data) keys. Serialization is performed recursively, and descends
    into the standard python container types (list, dict, tuple, set).

    Some of this code was adapted from these fantastic blog posts by Chris
    Wagner and Sunil Arora:

      http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html
      http://sunilarora.org/serializable-decorator-for-python-class/

    '''

    class ThunderSerializeableObjectWrapper(object):

        def __init__(self, *args, **kwargs):
            self.wrapped = cls(*args, **kwargs)

        # Allows transparent access to the attributes of the wrapped class
        def __getattr__(self, *args):
            if args[0] != 'wrapped':
                return getattr(self.wrapped, *args)
            else:
                return self.__dict__['wrapped']

        # Allows transparent access to the attributes of the wrapped class
        def __setattr__(self, *args):
            if args[0] != 'wrapped':
                return setattr(self.wrapped, *args)
            else:
                self.__dict__['wrapped'] = args[1]

        # Delegate to wrapped class for special python object-->string methods
        def __str__(self):
            return self.wrapped.__str__()
        def __repr__(self):
            return self.wrapped.__repr__()
        def __unicode__(self):
            return self.wrapped.__unicode__()

        # Delegate to wrapped class for special python methods
        def __call__(self, *args, **kwargs):
            return self.wrapped.__str__(*args, **kwargs)

        # ------------------------------------------------------------------------------
        # SERIALIZE()

        def serialize(self, numpyStorage='auto'):
            '''
            Serialize this object to a python dictionary that can easily be converted
            to/from JSON using Python's standard JSON library.

            Arguments

              numpy-storage: choose one of ['auto', 'ascii', 'base64'] (default: auto)

              Use the 'numpyStorage' argument to select whether numpy arrays
              will be encoded in ASCII (as a list of lists) in Base64 (i.e.
              space efficient binary), or to select automatically (the default)
              depending on the size of the array. Currently the Base64 encoding
              is selecting if the array has more than 1000 elements.

            Returns

              The object encoded as a python dictionary with "JSON-safe" datatypes that is ready to
              be converted to a string using Python's standard JSON library (or another library of
              your choice).
            '''
            from collections import namedtuple, Iterable, OrderedDict
            from numpy import ndarray

            def serializeRecursively(data):
                import datetime

                if data is None or isinstance(data, (bool, int, long, float, basestring)):
                    return data
                if isinstance(data, list):
                    return [serializeRecursively(val) for val in data]           # Recurse into lists
                if isinstance(data, OrderedDict):
                    return {"py/collections.OrderedDict":
                            [[serializeRecursively(k), serializeRecursively(v)] for k, v in data.iteritems()]}
                if _isNamedTuple(data):
                    return {"py/collections.namedtuple": {
                        "type":   type(data).__name__,
                        "fields": list(data._fields),
                        "values": [serializeRecursively(getattr(data, f)) for f in data._fields]}}
                if isinstance(data, dict):
                    if all(isinstance(k, basestring) for k in data):   # Recurse into dicts
                        return {k: serializeRecursively(v) for k, v in data.iteritems()}
                    else:
                        return {"py/dict": [[serializeRecursively(k), serializeRecursively(v)] for k, v in data.iteritems()]}
                if isinstance(data, tuple):                          # Recurse into tuples
                    return {"py/tuple": [serializeRecursively(val) for val in data]}
                if isinstance(data, set):                            # Recurse into sets
                    return {"py/set": [serializeRecursively(val) for val in data]}
                if isinstance(data, datetime.datetime):
                    return {"py/datetime": str(data)}
                if isinstance(data, complex):
                    return {"py/complex": [ data.real, data.imag] }
                if isinstance(data, ndarray):
                    if numpyStorage == 'ascii' or (numpyStorage == 'auto' and data.size < 1000):
                        return {"py/numpy.ndarray.ascii": {
                            "shape": data.shape,
                            "values": data.tolist(),
                            "dtype":  str(data.dtype)}}
                    else:
                        from base64 import b64encode
                        return {"py/numpy.ndarray.base64": {
                        "shape": data.shape,
                        "values": b64encode(data),
                        "dtype":  str(data.dtype)}}

                raise TypeError("Type %s not data-serializable" % type(data))

            # If this object has slots, we need to convert the slots to a dict before serializing them.
            if hasattr(cls, "__slots__"):
                slotDict = { key: self.wrapped.__getattribute__(key) for key in cls.__slots__ }
                return serializeRecursively(slotDict)

            # Otherwise, we handle the object as though it has a normal __dict__ containing its attributes.
            else:
                return serializeRecursively(self.wrapped.__dict__)

        # ------------------------------------------------------------------------------
        # DESERIALIZE()

        @staticmethod
        def deserialize(serializedDict):
            '''
            Restore the object that has been converted to a python dictionary using an @serializable
            class's serialize() method.

            Arguments

                serializedDict: a python dictionary returned by serialize()

            Returns:

                A reconstituted class instance
            '''

            def restoreRecursively(dct):
                '''
                This object hook helps to deserialize object encoded using the
                serialize() method above.
                '''
                from numpy import frombuffer, dtype, array
                from base64 import decodestring

                # First, check to see if this is an encoded entry
                dataKey = None
                if isinstance(dct, dict):
                    filteredKeys = filter(lambda x: "py/" in x, dct.keys())

                    # If there is just one key with a "py/" prefix, that is the dataKey!
                    if len(filteredKeys) == 1:
                        dataKey = filteredKeys[0]

                # If no data key is found, we assume the data needs no further decoding.
                if dataKey is None:
                    return dct

                # Otherwise, decode it!
                if "py/dict" == dataKey:
                    return dict(restoreRecursively(dct["py/dict"]))
                if "py/tuple" == dataKey:
                    return tuple(restoreRecursively(dct["py/tuple"]))
                if "py/set" == dataKey:
                    return set(restoreRecursively(dct["py/set"]))
                if "py/collections.namedtuple" == dataKey:
                    data = restoreRecursively(dct["py/collections.namedtuple"])
                    return namedtuple(data["type"], data["fields"])(*data["values"])
                if "py/collections.OrderedDict" == dataKey:
                    return OrderedDict(restoreRecursively(dct["py/collections.OrderedDict"]))
                if "py/datetime" == dataKey:
                    from dateutil import parser
                    return parser.parse(dct["py/datetime"])
                if "py/complex" == dataKey:
                    data = dct["py/complex"]
                    return complex( float(data[0]), float(data[1]) )
                if "py/numpy.ndarray.ascii" == dataKey:
                    data = dct["py/numpy.ndarray.ascii"]
                    return array(data["values"], dtype=data["dtype"])
                if "py/numpy.ndarray.base64" == dataKey:
                    data = dct["py/numpy.ndarray.base64"]
                    arr = frombuffer(decodestring(data["values"]), dtype(data["dtype"]))
                    return arr.reshape(data["shape"])

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
                    thawedObject.__setattr__(key, restoredDict[key])

            # Otherwise simply update the objects dictionary en masse
            else:
                thawedObject.__dict__ = restoredDict

            # Finally, we would like this re-hydrated object to also be @serializable, so we re-wrap it
            # in the ThunderSerializeableObjectWrapper using the same trick with __new__().
            rewrappedObject = ThunderSerializeableObjectWrapper.__new__(ThunderSerializeableObjectWrapper)
            rewrappedObject.__dict__['wrapped'] = thawedObject

            # Return the re-constituted class
            return rewrappedObject

    # End of decorator.  Return the wrapper class from inside this closure.
    return ThunderSerializeableObjectWrapper




