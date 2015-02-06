""" Useful decorators that are used throughout the library """


def _isNamedTuple(obj):
  """Heuristic check if an object is a namedtuple."""
  from collections import namedtuple
  return hasattr(obj, "_fields") \
     and hasattr(obj, "_asdict") \
     and callable(obj._asdict)


def serializable(cls):
    """
    The @serializable decorator can decorate any class to make it easy to store
    that class in a human readable JSON format and then recall it and recover
    the original object instance. Classes instances that are wrapped in this
    decorator gain the serialize() method, and the class also gains a
    deserialize() static method that can automatically "pickle" and "unpickle" a
    wide variety of objects.

    Note that this decorator is NOT designed to provide generalized pickling
    capabilities. Rather, it is designed to make it very easy to convert small
    classes containing model properties to a human and machine parsable format
    for later analysis or visualization.

    A key feature of the @serializable decorator is that it can "pickle" data
    types that are not normally supported by Python's stock JSON dump() and
    load() methods. Supported datatypes include: list, set, tuple, namedtuple,
    OrderedDict, datetime objects, numpy ndarrays, and dicts with non-string
    (but still data) keys. Serialization is performed recursively, and descends
    into the standard python container types (list, dict, tuple, set). Unicode
    strings are not currently supported.

    IMPORTANT NOTE: The object decorated with @serializable must store their
    attributes in __slots__ or a __dict__, but not both! For example, you cannot
    _directly_ wrap a namedtuple in the serializable decorator. However, you can
    decorate an object that has an attribute pointing at an namedtuple. Only the
    decorated object is subject to this restriction.

    Some of this code was adapted from these fantastic blog posts by Chris
    Wagner and Sunil Arora:

      http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html
      http://sunilarora.org/serializable-decorator-for-python-class/

    Examples
    --------

      @serializable
      class Visitor(object):
          def __init__(self, ipAddr = None, agent = None, referrer = None):
              self.ip = ipAddr
              self.ua = agent
              self.referrer= referrer
              self.time = datetime.datetime.now()

      origVisitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

      # Serialize the object
      pickledVisitor = origVisitor.serialize()

      # Restore object
      recovVisitor = Visitor.deserialize(pickledVisitor)


    """

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
                array. Currently the Base64 encoding is selecting if the array
                has more than 1000 elements.

            Returns
            -------

              The object encoded as a python dictionary with "JSON-safe" datatypes that is ready to
              be converted to a string using Python's standard JSON library (or another library of
              your choice).

            """
            from collections import OrderedDict
            from numpy import ndarray

            def serializeRecursively(data):
                import datetime

                dataType = type(data)
                if dataType in frozenset(type(None), bool, int, long, float, str):
                    return data
                elif dataType == list:
                    return [serializeRecursively(val) for val in data]           # Recurse into lists
                elif dataType == OrderedDict:
                    return {"py/collections.OrderedDict":
                            [[serializeRecursively(k), serializeRecursively(v)] for k, v in data.iteritems()]}
                elif _isNamedTuple(data):
                    return {"py/collections.namedtuple": {
                        "type":   dataType.__name__,
                        "fields": list(data._fields),
                        "values": [serializeRecursively(getattr(data, f)) for f in data._fields]}}
                elif dataType == dict:
                    if all(type(k) == str for k in data):   # Recurse into dicts
                        return {k: serializeRecursively(v) for k, v in data.iteritems()}
                    else:
                        return {"py/dict": [[serializeRecursively(k), serializeRecursively(v)] for k, v in data.iteritems()]}
                elif dataType == tuple:                          # Recurse into tuples
                    return {"py/tuple": [serializeRecursively(val) for val in data]}
                elif dataType == set:                            # Recurse into sets
                    return {"py/set": [serializeRecursively(val) for val in data]}
                elif dataType == datetime.datetime:
                    return {"py/datetime.datetime": str(data)}
                elif dataType == complex:
                    return {"py/complex": [ data.real, data.imag] }
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
                elif dataType == ThunderSerializeableObjectWrapper:
                    # nested serializable object
                    return {"py/ThunderSerializeableObjectWrapper": {
                        "clsName": cls.__name__,
                        "clsModule": cls.__module__,
                    }}

                raise TypeError("Type %s not data-serializable" % dataType)

            # Check for unsupported class.
            if hasattr(self.wrapped, "__slots__") and hasattr(self.wrapped, "__dict__"):
                raise TypeError("Cannot serialize a class that has attributes in both __slots__ and __dict__")

            # If this object has slots, we need to convert the slots to a dict before serializing them.
            if hasattr(cls, "__slots__"):
                slotDict = { key: self.wrapped.__getattribute__(key) for key in cls.__slots__ }
                return serializeRecursively(slotDict)

            # Otherwise, we handle the object as though it has a normal __dict__ containing its attributes.
            else:
                return serializeRecursively(self.wrapped.__dict__)


        @staticmethod
        def deserialize(serializedDict):
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
                    from collections import namedtuple
                    data = restoreRecursively(dct["py/collections.namedtuple"])
                    return namedtuple(data["type"], data["fields"])(*data["values"])
                if "py/collections.OrderedDict" == dataKey:
                    from collections import OrderedDict
                    return OrderedDict(restoreRecursively(dct["py/collections.OrderedDict"]))
                if "py/datetime.datetime" == dataKey:
                    from dateutil import parser
                    return parser.parse(dct["py/datetime.datetime"])
                if "py/complex" == dataKey:
                    data = dct["py/complex"]
                    return complex( float(data[0]), float(data[1]) )
                if "py/numpy.ndarray" == dataKey:
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

        def save(self, f, numpyStorage='auto'):
            """Serialize wrapped object to a JSON file.

            Parameters
            ----------
            f : filename or file handle
                The file to write to. A passed handle will be left open for further writing.
            """
            def saveImpl(fp, numpyStorage_):
                fp.write(self.serialize(numpyStorage=numpyStorage_))
            if isinstance(f, basestring):
                with open(f, 'w') as handle:
                    saveImpl(handle, numpyStorage)
            else:
                # assume f is a file
                saveImpl(f, numpyStorage)

        @staticmethod
        def load(f):
            def loadImpl(fp):
                jsonStr = fp.read()
                return cls.deserialize(jsonStr)

            if isinstance(f, basestring):
                with open(f, 'w') as handle:
                    return loadImpl(handle)
            else:
                # assume f is a file
                return loadImpl(f)

    # End of decorator.  Return the wrapper class from inside this closure.
    return ThunderSerializeableObjectWrapper




