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
    """
    def __init__(self, params):
        if isinstance(params, dict):
            params = [params]
        self.params = params

    def names(self):
        """ List the names of all parameters. """
        return [p['name'].encode('ascii') for p in self.params]

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
        elif not isinstance(names, list):
            names = [names]

        out = []
        for p in self.params:
            if p['name'] in names:
                out.append(p['value'])

        if len(out) < len(names):
            raise KeyError("Only found values for %g of %g named parameters" % (len(out), len(names)))

        return asarray(out).squeeze()
