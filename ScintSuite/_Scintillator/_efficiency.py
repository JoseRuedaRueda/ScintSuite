"""
Contains the methods to load and deal with the efficiency of the scintillators

In the future it will also contain the routines to calculate the efficiency
with models different from Birk

Jose Rueda: jrrueda@us.es

Introduced in version 0.4.2
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from ScintSuite._Paths import Path as p
from ScintSuite._Machine import machine
from scipy.interpolate import interp1d
paths = p(machine)

import logging
logger = logging.getLogger('ScintSuite.Scintillator')


# ------------------------------------------------------------------------------
# --- Reading routine
# ------------------------------------------------------------------------------
def read_scintillator_efficiency(file, verb: bool = False):
    """
    Load the efficiency of a scintillator

    Jose Rueda: jrrueda@us.es

    :param  file: full path to the file to be loaded
    :param  verb: if true, the comments in the file will be printed
    """
    # --- Factors
    factors = {
        'keV': 1.0,
        'MeV': 1000.0,
        'eV': 0.001
    }
    # Read the data
    n = int(np.loadtxt(file, skiprows=8, max_rows=1))

    energy, yi = \
        np.loadtxt(file, skiprows=9, max_rows=n, unpack=True)
    # Read the comments:
    with open(file) as f:
        dummy = f.readlines()
        comments = dummy[:8] + dummy[(9 + n):]
    # Find the units
    for j in comments:
        com, line = j.split('#')
        line = line.strip()
        if line.startswith(('Energy', 'energy')):
            # Go for the energy
            cosos = line.split('(')
            unit = cosos[1]
            unit = unit.split(')')[0]
            fact = factors[unit]
            # Go for the yield
            unitYield = cosos[-1]
            unitYield = unitYield.split(')')[0]
            break

    # Print if needed:
    if verb:
        for i in range(len(comments)):
            # Avoid the two last characters to skip the \n
            print(comments[i][:-2])

    # Transform into an xarray for better handling
    out = xr.DataArray(yi, dims=('E',), coords={'E': energy*fact})
    out.attrs['Comments'] = comments
    out['E'].attrs['units'] = 'keV'
    out['E'].attrs['long_name'] = 'Energy'
    out.attrs['units'] = unitYield
    out.attrs['long_name'] = 'Yield'
    return out


# ------------------------------------------------------------------------------
# ---- Main class with efficiency
# ------------------------------------------------------------------------------
class ScintillatorEfficiency:
    """Class containing the scintillator efficiency."""

    def __init__(self, material: str = 'TgGreenA', particle: str = 'D',
                 thickness: int = 9,  verbose: bool = False):
        """
        Load data.

        :param  material: material of the plate
        :param  particle: particle which is being launch
        :param  thickness: thickness of the scintillator power

        :Notes:
        - The particle, material and thickness will be used to deduce the name
          of the fille to load: `<material>/<particle>_<thickness>.dat`
        """
        file = os.path.join(paths.ScintSuite, 'Data', 'ScintillatorEfficiency',
                            material, particle + '_' + str(thickness) + '.dat')
        ## Efficiency data:
        self.data = read_scintillator_efficiency(file, verbose)
        logger.info('Reading efficiency file: ' + file)

    def __call__(self, E, kwargs: dict = {}):
        """
        Wrap the interpolator.

        By default, extrapolation is selected
        :param E: input energy, in keV
        :param kwargs: optional arguments for the interpolator
        :return: interpolated vallues
        """
        kwargs2 = {'fill_value': 'extrapolate'}
        kwargs2.update(kwargs)
        return self.data.interp(E=E, kwargs=kwargs2)
