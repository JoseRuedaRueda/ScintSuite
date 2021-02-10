"""
Input/output library

Contains a miscellany of routines related with the different diagnostics, for
example the routine to read the scintillator efficiency files, common for all
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import netcdf
from version_suite import version
from LibPaths import Path
import os

paths = Path()


def read_scintillator_efficiency(file, plot: bool = False, verb: bool = True):
    """
    Load the efficiency of a scintillator

    Jose Rueda: jrrueda@us.es

    @param file: full path to the file to be loaded
    @param plot: is true, a quick plot will be done
    @param verb: if true, the comments in the file will be printed
    """
    # Read the data
    out = {'n': int(np.loadtxt(file, skiprows=8, max_rows=1))}

    out['energy'], out['yield'] = \
        np.loadtxt(file, skiprows=9, max_rows=out['n'], unpack=True)

    # Read the comments:
    with open(file) as f:
        dummy = f.readlines()
        out['comments'] = dummy[:8] + dummy[(9 + out['n']):]

    # Print if needed:
    if verb:
        for i in range(len(out['comments'])):
            # Avoid the two last characters to skip the \n
            print(out['comments'][i][:-2])

    # Plot if desired:
    if plot:
        fig, ax = plt.subplots()
        ax.plot(out['energy'], out['yield'])
        ax.set_xlabel('Energy [eV]')
        ax.set_xscale('log')
        ax.set_ylabel('Yield [photons / ion]')
        plt.tight_layout()
    return out


# -----------------------------------------------------------------------------
# --- Tomography
# -----------------------------------------------------------------------------
def save_FILD_W(W4D, grid_p, grid_s, W2D=None, filename: str = None,
                efficiency: bool = False):
    """
    Save the FILD_W to a .netcdf file

    Jose rueda: jrrueda@us.es

    @todo: include the units of W

    @param W4D: 4D Weight matrix to be saved
    @param grid_p: grid at the pinhole
    @param grid_s: grid at the scintillator
    @param W2D: optional, 2D contraction of W4D
    @param filename: Optinal filename to use, if none, it will be saved at the
    results file with the name W_FILD_<date, time>.nc
    @param efficiency: bool to save at the file, to indicate if the efficiency
    was used in the calculation of W
    """
    print('.... . .-.. .-.. ---')
    if filename is None:
        a = time.localtime()
        name = 'W_FILD_' + str(a.tm_year) + '_' + str(a.tm_mon) + '_' +\
            str(a.tm_mon) + '_' + str(a.tm_hour) + '_' + str(a.tm_min) +\
            '.nc'
        filename = os.path.join(paths.Results, name)
    print('Saving results in: ', filename)
    with netcdf.netcdf_file(name, 'w') as f:
        f.history = 'Done with version ' + version

        # --- Save the pinhole grid
        f.createDimension('number', 1)
        nr_pin = f.createVariable('nr_pin', 'i', ('number', ))
        nr_pin[:] = grid_p['nr']
        nr_pin.units = ' '
        nr_pin.long_name = 'Number of points in r, pinhole'

        np_pin = f.createVariable('np_pin', 'i', ('number', ))
        np_pin[:] = grid_p['np']
        np_pin.units = ' '
        np_pin.long_name = 'Number of points in pitch, pinhole'

        f.createDimension('np_pin', grid_p['np'])
        p_pin = f.createVariable('p_pin', 'float64', ('np_pin', ))
        p_pin[:] = grid_p['p']
        p_pin.units = 'degrees'
        p_pin.long_name = 'Pitch values, pinhole'

        f.createDimension('nr_pin', grid_p['nr'])
        r_pin = f.createVariable('r_pin', 'float64', ('nr_pin', ))
        r_pin[:] = grid_p['r']
        r_pin.units = 'cm'
        r_pin.long_name = 'Gyroradius values, pinhole'

        # --- Save the scintillator grid
        nr_scint = f.createVariable('nr_scint', 'i', ('number', ))
        nr_scint[:] = grid_s['nr']
        nr_scint.units = ' '
        nr_scint.long_name = 'Number of points in r, scint'

        np_scint = f.createVariable('np_scint', 'i', ('number', ))
        np_scint[:] = grid_s['np']
        np_scint.units = ' '
        np_scint.long_name = 'Number of points in pitch, scint'

        f.createDimension('np_scint', grid_s['np'])
        p_scint = f.createVariable('p_scint', 'float64', ('np_scint', ))
        p_scint[:] = grid_s['p']
        p_scint.units = 'degrees'
        p_scint.long_name = 'Pitch values, scint'

        f.createDimension('nr_scint', grid_s['nr'])
        r_scint = f.createVariable('r_scint', 'float64', ('nr_scint', ))
        r_scint[:] = grid_s['r']
        r_scint.units = 'cm'
        r_scint.long_name = 'Gyroradius values, scint'

        # Save the 4D W
        W = f.createVariable('W4D', 'float64',
                             ('nr_scint', 'np_scint', 'nr_pin', 'np_pin'))
        W[:, :, :, :] = W4D
        W.units = 'a.u.'
        W.long_name = 'Intrument function'

        if W2D is not None:
            f.createDimension('n_scint', grid_s['nr'] * grid_s['np'])
            f.createDimension('n_pin', grid_p['nr'] * grid_p['np'])
            W2 = f.createVariable('W2D', 'float64', ('n_scint', 'n_pin'))
            W2[:, :] = W2D
            W2.units = 'a.u.'
            W2.long_name = 'Instrument funtion ordered in 2D'

        print('Make sure you are setting the efficiency flag properly!!!')
        if efficiency:
            eff = f.createVariable('efficiency', 'i', ('number',))
            eff[:] = int(efficiency)
            eff.units = ' '
            eff.long_name = '1 Means efficiency was activated to calculate W'
    print('-... -.-- . / -... -.-- .')
