"""
Basic variable class

Jose Rueda Rueda: jrrueda@us.es
"""
import numpy as np


class BasicVariable():
    """
    Simple class to contain the data of a given variable, with metadata

    Jose Rueda Rueda: jrrueda@us.es
    """

    def __init__(self, name: str = None, units: str = None,
                 data: np.ndarray = None):
        """
        Just store everything on place

        @param name: Name atribute
        @param units: Physical units of the variable
        @param data: array with the data values
        """
        self.name = name
        self.units = units
        self.data = data

        # try to get the shape of the data
        try:
            self.shape = data.shape
            self.size = data.size
        except AttributeError:
            self.shape = None
            self.size = None

        # Deduce the label for plotting
        if (name is not None) and (units is not None):
            self.plot_label = '%s [%s]' % (name.lower().capitalize(), units)
        else:
            self.plot_label = None
