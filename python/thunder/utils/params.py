"""
Class for working with auxillery parameters
"""

from numpy import asarray


class Params(object):
    """
    Store simple parameters or lists of parameters.

    Assumes parameters are either a dictionary or a list of dictionaries,
    where each dictionary has a "name" field with a string,
    a "value" field, and potentially other optional fields.

    Attributes
    ----------
    params : list of dicts
        List of dictionaries each containing a parameter, where each
        parameter has at least a "name" field and a "value" field
    """
    def __init__(self, params):
        if isinstance(params, dict):
            params = [params]
        self._params = params

    def __getitem__(self, names=None):
        return self.values(names)

    def __repr__(self):
        s = self.__class__.__name__ + '\n'
        s += 'names: ' + str(self.names())
        return s

    def names(self):
        """
        List the names of all parameters.
        """
        return [p['name'].encode('ascii') for p in self._params]

    def values(self, names=None):
        """
        Return values from parameters, optionally given a set of names.

        Parameters
        ----------
        names : str or list, optional, default=None
            The names to retrieve values for.
        """
        if names is None:
            names = self.names()
        if type(names) is str:
            names = [names]
        elif not isinstance(names, list):
            names = list(names)

        out = []
        for p in self._params:
            if p['name'] in names:
                out.append(p['value'])

        if len(out) < len(names):
            raise KeyError("Only found values for %g of %g named parameters" % (len(out), len(names)))

        return asarray(out).squeeze()
