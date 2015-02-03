""" Useful decorators that are used throughout the library """

def _isnamedtuple(obj):
  """Heuristic check if an object is a namedtuple."""
  return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def serializable(cls):
    '''
    The @serializable decorator can be added to a class to make it easy to store
    in a human readable JSON format. Classes instances that are wrapped in this
    decorator gain the serialize() method, and the class also gains a
    deserialize() static method that can automatically "pickle" and "unpickle"
    a wide variety of objects.

    Usage example:

      @serializable
      class Visitor():
          def __init__(self, ip_addr = None, agent = None, referrer = None):
              self.ip = ip_addr
              self.ua = agent
              self.referrer= referrer
              self.time = datetime.datetime.now()

      orig_visitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

      #serialize the object
      pickled_visitor = orig_visitor.serialize()

      #restore object
      recov_visitor = Visitor.deserialize(pickled_visitor)

    Note that this decorator is NOT designed to provide generalized pickling
    capabilities. Rather, it is designed to make it very easy to convert small
    classes containing model properties to a human and machine parsable format
    for later analysis or visualization.

    Any @serializable class can contain data that are not normally supported by
    Python's stock JSON dump() and load() methods. Supported datatypes include
    list, set, tuple, namedtuple, OrderedDict, datetime objects, numpy ndarrays,
    and dicts with non-string (but still data) keys.

    Serialization is performed recursively, and descends into the standard
    python container types (list, dict, tuple, set).

    Some of this code was posted in this fantastic blog post by Chris Wagner:

      http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html

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

        def serialize(self, numpy_storage='auto'):
            '''
            Serialize this object to a python dictionary that can easily be converted
            to/from JSON using Python's standard JSON library.

            Arguments

              numpy-storage: choose one of ['auto', 'ascii', 'base64'] (default: auto)

              Use the 'nmupy_storage' argument to select whether numpy arrays
              will be encoded in ASCII (as a list of lists) in Base64 (i.e.
              space efficient binary), or to select automatically (the default)
              depending on the size of the array. Currently the Base64 encoding
              is selecting if the array has more than 1000 elements.

            Returns

              The object encoded as a python dictionary with "JSON-safe" datatypes that is ready to
              be converted to a string using Python's standard JSON library (or another library of
              your choice.
            '''
            from collections import namedtuple, Iterable, OrderedDict
            import numpy as np

            def serialize_recursively(data):
                import datetime

                if data is None or isinstance(data, (bool, int, long, float, basestring)):
                    return data
                if isinstance(data, list):
                    return [serialize_recursively(val) for val in data]           # Recurse into lists
                if isinstance(data, OrderedDict):
                    return {"py/collections.OrderedDict":
                            [[serialize_recursively(k), serialize_recursively(v)] for k, v in data.iteritems()]}
                if _isnamedtuple(data):
                    return {"py/collections.namedtuple": {
                        "type":   type(data).__name__,
                        "fields": list(data._fields),
                        "values": [serialize_recursively(getattr(data, f)) for f in data._fields]}}
                if isinstance(data, dict):
                    if all(isinstance(k, basestring) for k in data):   # Recurse into dicts
                        return {k: serialize_recursively(v) for k, v in data.iteritems()}
                    else:
                        return {"py/dict": [[serialize_recursively(k), serialize_recursively(v)] for k, v in data.iteritems()]}
                if isinstance(data, tuple):                          # Recurse into tuples
                    return {"py/tuple": [serialize_recursively(val) for val in data]}
                if isinstance(data, set):                            # Recurse into sets
                    return {"py/set": [serialize_recursively(val) for val in data]}
                if isinstance(data, datetime.datetime):
                    return {"py/datetime": str(data)}
                if isinstance(data, np.ndarray):
                    if numpy_storage == 'ascii' or (numpy_storage == 'auto' and data.size < 1000):
                        return {"py/numpy.ndarray.ascii": {
                            "shape": data.shape,
                            "values": data.tolist(),
                            "dtype":  str(data.dtype)}}
                    else:
                        import base64
                    return {"py/numpy.ndarray.base64": {
                        "shape": data.shape,
                        "values": base64.b64encode(data),
                        "dtype":  str(data.dtype)}}

                raise TypeError("Type %s not data-serializable" % type(data))

            # Start serializing from the top level object dictionary
            return serialize_recursively(self.wrapped.__dict__)

        # ------------------------------------------------------------------------------
        # DESERIALIZE()

        @staticmethod
        def deserialize(serialized_dict):
            '''
            Restore the object that has been converted to a python dictionary using an @serializable
            class's serialize() method.

            Arguments

                serialized_dict: a python dictionary returned by serialize()

            Returns:

                A reconstituted class instance
            '''

            def restore_recursively(dct):
                '''
                This object hook helps to deserialize object encoded using the
                serialize() method above.
                '''
                import numpy as np
                import base64

                if "py/dict" in dct:
                    return dict(restore_recursively(dct["py/dict"]))
                if "py/tuple" in dct:
                    return tuple(restore_recursively(dct["py/tuple"]))
                if "py/set" in dct:
                    return set(restore_recursively(dct["py/set"]))
                if "py/collections.namedtuple" in dct:
                    data = restore_recursively(dct["py/collections.namedtuple"])
                    return namedtuple(data["type"], data["fields"])(*data["values"])
                if "py/collections.OrderedDict" in dct:
                    return OrderedDict(restore_recursively(dct["py/collections.OrderedDict"]))
                if "py/datetime" in dct:
                    from dateutil import parser
                    return parser.parse(dct["py/datetime"])
                if "py/numpy.ndarray.ascii" in dct:
                    data = dct["py/numpy.ndarray.ascii"]
                    return np.array(data["values"], dtype=data["dtype"])
                if "py/numpy.ndarray.base64" in dct:
                    data = dct["py/numpy.ndarray.base64"]
                    arr = np.frombuffer(base64.decodestring(data["values"]), np.dtype(data["dtype"]))
                    return arr.reshape(data["shape"])

                # Base case: data type needs no further decoding.
                return dct

            # First we must restore the object's dictionary entries.  These are decoded recursively
            # using the helper function above.
            restored_dict = {}
            for k in serialized_dict.keys():
                restored_dict[k] = restore_recursively(serialized_dict[k])

            # Next we recreate the object. Calling the __new__() function here creates
            # an empty object without calling __init__(). We then take this empty
            # shell of an object, and set its dictionary to the reconstructed
            # dictionary we pulled from the JSON file.
            thawed_object = cls.__new__(cls)
            thawed_object.__dict__ = restored_dict

            # Finally, we would like this re-hydrated object to also be @serializable, so we re-wrap it
            # in the ThunderSerializeableObjectWrapper using the same trick with __new__().
            rewrapped_object = ThunderSerializeableObjectWrapper.__new__(ThunderSerializeableObjectWrapper)
            rewrapped_object.__dict__['wrapped'] = thawed_object

            # Return the re-constituted class
            return rewrapped_object

    # End of decorator.  Return the wrapper class from inside this closure.
    return ThunderSerializeableObjectWrapper




