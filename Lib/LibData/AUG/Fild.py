"""
Contain FILD object.

Jose Rueda: jrrueda@us.es
"""

import os
import f90nml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)


# --- Default parameters:
file = os.path.join(paths.ScintSuite, 'Data', 'Calibrations', 'FILD', 'AUG',
                    'DefaultParameters.txt')
default = f90nml.read(file)['orientation']


# --- Auxiliar routines to load and plot FILD4 trajectory:
def load_FILD4_trajectory(shot, path=paths.FILD4_trayectories):
    """
    Load FILD4 trayectory

    Jose Rueda: jrrueda@us.es

    Note: This is a temporal function, in the future will be replaced by one to
    load trayectories from shotfiles

    @param shot: Shot number to load
    @param path: Path to the main folder with FILD4 trajectories
    """
    # --- Load the power supply output data
    shot_str = str(shot)
    try:
        file = os.path.join(path, 'output_raw', shot_str[0:2],
                            'FILD_MDRS_' + shot_str + '.txt')
        print('Looking for file: ', file)
        data = np.loadtxt(file, skiprows=1)
        # Delete the last line of the data because is always zero
        dat = np.delete(data, -1, axis=0)
        # Delete points where PS output is zero. This **** instead of giving
        # as ouput the points where the trajectory was requested, it always
        # gives as ouput a given number of rows, and set to zero the non used
        # ones...
        fi = dat[:, 2] < 1
        fv = dat[:, 4] < 1
        flags = (fv * fi).astype(np.bool)
        PSouput = {
            'V_t_obj': dat[~flags, 0] / 1000.0,
            'V_obj': dat[~flags, 1],
            'I_t': dat[~flags, 2] * 1.0e-9,
            'I': dat[~flags, 3],
            'V_t': dat[~flags, 4] * 1.0e-9,
            'V': dat[~flags, 5]
        }
    except OSError:
        print('File with power supply outputs not found')
        PSouput = None
    # --- Load the reconstructed trajectory
    R4_lim = [2.082410, 2.015961]    # R min and max of FILD4
    Z4_lim = [-0.437220, -0.437906]  # z min and max of FILD4
    ins_lim = [0, 0.066526]          # Maximum insertion
    try:
        file = os.path.join(path, 'output_processed', shot_str[0:2],
                            shot_str + '.txt')
        print('Looking for file: ', file)
        data = np.loadtxt(file, skiprows=2, delimiter=',')
        insertion = {
            't': data[:, 0],
            'insertion': data[:, 1],
        }
        if data[:, 1].max() > ins_lim[1]:
            warnings.warn('FILD4 insertion larger than the maximum!!!')
        position = {
            't': data[:, 0],
            'R': R4_lim[0] + (data[:, 1]-ins_lim[0])/(ins_lim[0]-ins_lim[1])
            * (R4_lim[0]-R4_lim[1]),
            'z': Z4_lim[0]+(data[:, 1]-ins_lim[0])/(ins_lim[0]-ins_lim[1])
            * (Z4_lim[0]-Z4_lim[1])
        }
    except OSError:
        print('File with trajectory not found')
        insertion = None

    return {'PSouput': PSouput, 'insertion': insertion, 'position': position}


def plot_FILD4_trajectory(shot, PS_output=False, ax=None, ax_PS=None,
                          line_params={}, line_params_PS={}, overlay=False,
                          unit='cm'):
    """
    Plot FILD4 trayectory

    Jose Rueda: jrrueda@us.es

    Note: this is in beta phase, improvement suggestions are wellcome

    @param shot: shot you want to plot
    @param PS_output: flag to plot the output of the power supply
    @param ax: axes where to plot the trajectory. If none, new axis will be
               created
    @param ax_PS: Array of two axes where we want to plot the PS data. ax_PS[0]
                  will be for the voltaje while ax_PS[1] for the intensity. If
                  None, new axis  will be created
    @param line_params: Line parameters for the trajectory plotting
    @param line_params_PS: Line parameters for the PS plots. Note: same dict
                           will be used for the Voltaje and intensity plots, be
                           carefull if you select the 'color'
    @param overlay: Flag to overlay the trayectory over the current plot. The
                    insertion will be plotted in arbitrary units on top of it.
                    ax input is mandatory for this
    """
    line_options = {
        'label': '#' + str(shot),
    }
    line_options.update(line_params)
    # ---
    factor = {
        'cm': 100.0,
        'm': 1.0,
        'inch': 100.0 / 2.54,
        'mm': 1000.0
    }
    # --- Load the position
    position = load_FILD4_trajectory(shot)

    # --- Plot the position
    if ax is None:
        fig, ax = plt.subplots()
    if overlay:  # Overlay the trayectory in an existing plot:
        print('Sorry, still not implemented')
    else:
        ax.plot(position['insertion']['t'],
                factor[unit] * position['insertion']['insertion'],
                **line_options)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Insertion [' + unit + ']')
        ax.set_xlim(0, 1.1 * position['insertion']['t'].max())
        ymax = 1.1 * factor[unit] * position['insertion']['insertion'].max()
        ax.set_ylim(0, ymax)
        ax.legend()

    # --- Plot the PS output
    if PS_output:
        if ax_PS is None:
            fig2, ax_PS = plt.subplots(2, 1, sharex=True)
        # Plot the voltage
        ax_PS[0].plot(position['PSouput']['V_t_obj'],
                      position['PSouput']['V_obj'],
                      label='Objective')
        ax_PS[0].plot(position['PSouput']['V_t'],
                      position['PSouput']['V'],
                      label='Real')
        ax_PS[0].set_ylabel('Voltage [V]')
        ax_PS[0].legend()
        ax_PS[1].plot(position['PSouput']['I_t'],
                      position['PSouput']['I'])
        ax_PS[1].set_ylabel('Intensity [A]')
    plt.show()


# --- FILD object
class FILD_logbook:
    """
    Contain all geometrical parameters and path information of FILD

    Jose Rueda - jrrueda@us.es

    Introduced in version 0.7.2
    """

    def __init__(self, shot: int = 39612, id: int = 1,
                 file: str = paths.FILDPositionDatabase,
                 verbose: bool = True):
        """
        Initialise the object

        @param: shot, integer, shot number
        @param: id, fild number
        """
        self.shot = int(shot)
        self.id = int(id)
        # Load the database as a pandas frame
        if verbose:
            print('Looking for file: ', file)
        dummy = pd.read_excel(file, engine='openpyxl', header=[0, 1])
        self.database = dummy['FILD'+str(id)].copy()
        self.database['shot'] = dummy.Shot.Number.values.astype(int)
        if shot not in self.database['shot'].values:
            warnings.warn('Shot not found in the database, do it manually')
        else:
            self.position, self.orientation = self._get_coordinates()

    def _get_coordinates(self):
        """
        Get the position of the FILD detector.

        Jose Rueda - jrrueda@us.es

        @param shot: shot number to look in the database
        """
        # Get the shot index in the database
        i, = np.where(self.database['shot'].values == self.shot)[0]
        # --- Get the postion
        position = {        # Initialise the position
            'R': 0.0,
            'z': 0.0,
            'phi': 0.0,
        }
        if self.id != 4:
            if 'R [m]' in self.database.keys():  # Look for R in the database
                position['R'] = self.database['R [m]'].values[i]
            else:  # Take the default approx value
                position['R'] = default['r'][self.id-1]
            if 'Z [m]' in self.database.keys():  # Look for Z in the database
                position['z'] = self.database['Z [m]'].values[i]
            else:  # Take the default approx value
                position['z'] = default['z'][self.id-1]
            if 'Phi [deg]' in self.database.keys():  # Look for phi in the db
                position['phi'] = self.database['Phi [deg]'].values[i]
            else:  # Take the default approx value
                position['phi'] = default['phi'][self.id-1]
        else:  # We have FILD4, the movable FILD
            # Ideally, we will have an optic calibration database which is
            # position dependent, therefore we should keep all the FILD
            # trayectory.
            # However, this calibration database was not given by the previous
            # operator of the 'in-shot' movable FILD, neither any details of
            # the optical design which could allow us to create it. So until we
            # dismount and examine the diagnostic piece by piece, this
            # trayectory is irrelevant, we will just keep the average position,
            # which is indeed 'okeish', as the resolution of this FILD in AUG
            # is poor, so a small missalignement will not be noticed.
            # To calculate this average position, I will take the average of
            # the positions at least 5 mm further from the minimum limit
            try:
                dummy = load_FILD4_trajectory(self.shot)
                min = dummy['position']['R'].min()
                flags = dummy['position']['R'] > (min + 0.005)
                position['R'] = dummy['position']['R'][flags].mean()
                position['z'] = dummy['position']['z'][flags].mean()
            except OSError:    # Shot not found in the database
                position['R'] = default['r'][self.id-1]
                position['z'] = default['z'][self.id-1]
            # FILD is always the same:
            position['phi'] = default['phi'][self.id-1]
        # --- Get the orientation:
        # In the case of AUG, the alpha and beta angles are fixed by
        # construction they can't be changed therefore we will not even look in
        # in the logbook.
        orientation = {
            'alpha': default['alpha'][self.id-1],
            'beta': default['beta'][self.id-1],
            'psi': default['psi'][self.id-1]
        }
        return position, orientation

    def reshot(self, shot):
        """
        Get another shot.

        Jose Rueda - jrrueda@us.es

        Change the position and orientation for the values of a new shot

        @param shot: new shot number
        """
        self.shot = int(shot)
        self.position, self.orientation = self._get_coordinates()
