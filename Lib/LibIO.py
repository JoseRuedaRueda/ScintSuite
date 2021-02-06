"""
Input/output library

Contains a miscellany of routines related with the different diagnostics, for
example the routine to read the scintillator efficiency files, common for all
"""

import numpy as np
import matplotlib.pyplot as plt


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
