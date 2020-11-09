"""@package LibDataAUG
Module containing the routines to interact with the AUG database
"""

import sys
## todo a system path to AFS is hard written here
# Path to the IPP python library
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
# Module to load the equilibrium
from sf2equ_20200525 import EQU
# Module to map the equilibrium
import mapeq_20200507 as meq
# Module to load vessel components
import get_gc
# Other libraries
import numpy as np
# import matplotlib.pyplot as plt




def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, tiniEQU: float = None, tendEQU: float = None,
                  time: float = None, equ=None):
    """
    Wrapper to get AUG magnetic field

    Jose Rueda: jose.rueda@ipp.mpg.de

    Adapted from: https://www.aug.ipp.mpg.de/aug/manuals/map_equ/

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param tiniEQU: Initial time to load the equilibrium object, only valid
    if equ is not pass as input (in s)
    @param tendEQU: End time to load the equilibrium object, only valid if
    equ is not pass as input (in s)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object of clas EQU (see
    https://www.aug.ipp.mpg.de/aug/manuals/map_equ/equ/html/classsf2equ__20200525_1_1EQU.html)
    @return br: Radial magnetic field (nt, nrz_in), [T]
    @return bz: z magnetic field (nt, nrz_in), [T]
    @return bt: toroidal magnetic field (nt, nrz_in), [T]
    @return bp: poloidal magnetic field (nt, nrz_in), [T]
    """
    # If the equilibrium object is not an input, let create it
    if equ is None:
        equ = EQU(shot, diag=diag, exp=exp, ed = ed, tbeg=tiniEQU,
                  tend=tendEQU)
    # If the equilibrium object is an input, check if the shot number is correct
    else:
        if equ.shot != shot:
            print('Shot number of the received equilibrium does not match!')
            br = 0
            bz = 0
            bt = 0
            bp = 0
            return br, bz, bt, bp
    # Now calculate the field
    br, bz, bt = meq.rz2brzt(equ, r_in=Rin, z_in=zin, t_in=time)
    bp = np.hypot(br, bz)
    return br, bz, bt, bp


def plot_vessel(ax, projection: str = 'poloidal', line_properties: dict = {},
                nshot: int = 30585):
    """
    Plot AUG vessel

    José Rueda Rueda

    Poloidal plot of the vessel is directly extracted from IPP tutorial:
    https://www.aug.ipp.mpg.de/aug/manuals/map_equ/

    @param ax: axes where to plot
    @param projection: 'poloidal' or 'toroidal'
    @param line_properties: dictionary with the argument for the function
    plot of matplat lib (example, color, linewidth...)
    @return: Vessel plotted in the selected axes
    """
    ## todo plot toroidal vessel
    # Make sure that the color property is in the line_properties option,
    # if not, python will plot every part of the vessel in a different color
    # and we will have a funny output...
    if 'color' not in line_properties:
        line_properties['color'] = 'k'

    if projection == 'poloidal':
        # Get vessel coordinates
        gc_r, gc_z = get_gc.get_gc(nshot)
        for key in gc_r.keys():
            # print(key)
            ax.plot(gc_r[key], gc_z[key], **line_properties)
    elif projection == 'toroidal':
        print('Sorry, this option is not jet implemented. Talk to Jose Rueda')
    else:
        print('Not recognised argument')

    return