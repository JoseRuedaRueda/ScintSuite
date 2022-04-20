"""Class to interact with the npa cx output file"""
import os
import numpy as np
import matplotlib.pyplot as plt
from Lib.SimulationCodes.FIDASIM._read import read_npa


class NPA():
    """
    NPA FIDASIM markers

    Jose Rueda: jrrueda@us.es
    """

    def __init__(self, filename, A=2.014, verbose: bool = True):
        """
        Initalise the class

        @param filename: filename to be read
        @param verbose: bollean flag to print some info in the console
        """
        if not os.path.isfile(filename):
            raise Exception('File not found')
        self._dat = read_npa(filename, verbose)
        # Get the energy of these guys in keV:
        self._dat['energy'] = 0.5 * (self._dat['v']**2).sum(axis=1) * A \
            * 1.660538782e-27 / 1.602176487e-19 / 1000.0/1.0e4
        self._dat['Ri'] = np.sqrt(self._dat['ipos'][:, 0]**2
                                  + self._dat['ipos'][:, 1]**2)
        # Print the information
        if verbose:
            print('shot: %i' % self._dat['shot'])
            print('t: %.3f s' % self._dat['time'])
            print('# of markers: %i' % self._dat['counter'])

    def plot1Dhistogram(self, var, bins=60, includeW=True, ax=None,
                        line_params: dict = {}, normalise: bool = True,
                        var_limit: str = None, limits: tuple = None):
        """
        Plot a 1D histogram of a given variable

        @param var: variable we want to plot
        @param bins: number of bins or bin edges
        @param includeW: flag to include the weight of the markers
        @param ax: axes where to plot
        @param line_params: dict containing the line parameters for the plot
        @param normalise: if true, normalise y axis to one
        @param var_limit: name of the variable to restric the markers
        @param limits: tuple containing minimum and maximum values of the
            control variable

        @return xcenter: x axis of the histogram
        @return H: the plotted histogram
        """
        created = True
        if ax is None:
            fig, ax = plt.subplots()
        if includeW:
            w = self._dat['wght']
        else:
            w = np.ones(self._dat['wght'].size)
        if var_limit is not None:
            flags = (self._dat[var_limit] < limits[1]) \
                * (self._dat[var_limit] > limits[0])
        else:
            flags = np.ones(w.shape, dtype=bool)
        H, xedges = np.histogram(self._dat[var][flags], bins=bins,
                                 weights=w[flags])
        xcenter = 0.5*(xedges[:-1] + xedges[1:])
        if normalise:
            H /= H.max()
        ax.plot(xcenter, H, **line_params)
