"""
Routines to load and work with FILD trajectories.

FILD4 presents some uncertainties on its position that imposibilitates a
routinary reconstruction of its trajectory. Therefore, this is an "artisan"
process. For further info, contact Javier Hidalgo (jhsalaverri@us.es).

Due to an issue with the FILD4 computer, the trajectories are stored in
Javier Hidalgo's computer (javih). If you cannot access the required folders
(found in the path) contact him.

Javier Hidalgo-Salaverri: jhsalaverri@us.es
"""

import os
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol
import aug_sfutils as sf
import Lib.LibData.AUG.DiagParam as params
import Lib as ss
import f90nml
import Lib.errors as errors
from Lib._Paths import Path
from Lib._Machine import machine
paths = Path(machine)

try:
    import shapely.geometry as geom
except ModuleNotFoundError:
    pass


def get_dist2sep(shot: int = None, R: float = None, z: float = None,
                 t: float = None, diag: str = 'EQH', plot_sep: bool = True,
                 plot_dist: bool = True):
    """
    Get the distance to the separatrix during a shot.

    :param  shot
    :param  R: R array [m]
    :param  z: z array [m]
    :param  t: time array. If None, entire shot
    :param  diag: diagnostic for equilibrium reconstruction
    :param  plot_sep: plot separatrix position
    :param  plot_dist: plot distance to the separatrix

    returns distance to the separatrix in m
    """

    # Load the separatrix
    equ = sf.EQU(shot, diag=diag)
    r_sep, z_sep = sf.rho2rz(equ, rho_in=1.0, t_in=t,
                             coord_in='rho_pol')
    if t is None:
        R = R*np.ones(r_sep.shape)
        z = z*np.ones(r_sep.shape)
        t = equ.time

    # Get the minimum distance between the probe head and the separatrix
    dist_sep = np.zeros(r_sep.shape)
    for i in range(len(r_sep)):
        line_coords = np.transpose(np.array([r_sep[i][0], z_sep[i][0]]))
        line = geom.LineString(line_coords)
        point = geom.Point([R[i], z[i]])
        dist_sep[i] = point.distance(line)  # in m

    if plot_sep:
        fig, ax = plt.subplots()
        title = '#'+str(shot)+'\n t = ' + str(t[0])+' - '+str(t[-1])+' s'
        fig.suptitle(title)
        plt.grid()
        plt.axis('equal')
        for i in range(len(t)):
            ax.plot(np.array(r_sep[i]).squeeze(),
                    np.array(z_sep[i]).squeeze(), '-b')
        ax.plot(R, z, 'xr')
        plt.show()

    if plot_dist:
        fig2, ax2 = plt.subplots(2, sharex = True)
        title = '#'+str(shot)
        fig2.suptitle(title)
        ax2[0].plot(t, R)
        ax2[0].set_ylabel('R [m]')
        ax2[1].plot(t, dist_sep)
        ax2[1].set_xlabel('Time [s]')
        ax2[1].set_ylabel('Dist2sep [m]')
        plt.show()
    return dist_sep

def FILD2sep(shot: int, geomID: int,  insertion: float = None, t: float = None, 
             diag: str = 'EQH', plot_sep: bool = True, plot_dist: bool = True):
    """
    Get the distance of a certain FILD to the separatrix. If no insertion is 
    given, the logbook value is used.
    
    :param  shot
    :param  geomID: collimator ID
    @insertion: in manipulator units [mm]
    :param  t: time array. If None, entire shot
    :param  diag: diagnostic for equilibrium reconstruction
    :param  plot_sep: plot separatrix position
    :param  plot_dist: plot distance to the separatrix

    returns distance to the separatrix in m
    """
    geomID = geomID.lower()
    if insertion is None:
        lb = ss.dat.FILD_logbook()
        print('Logbook values are taken:')
        # Get the used collimator geometries in that shot
        col_geom = []
        filds = []
        for i in range(1,6):
            try:
                col_geom.append(lb.getGeomID(shot = shot, FILDid = i).lower())
                filds.append(i)
            except:
                pass
        # Get the position if the geomety was used
        if geomID in col_geom:
            fildID = filds[col_geom == geomID]
            position = lb.getPosition(shot = shot, FILDid = fildID)
            print('\nFILD'+str(fildID)+' ('+geomID.upper()+') #'+str(shot)+':')
            print(position)
            print('\n')
        else:
            raise errors.NotFoundGeomID(geomID+ ' was not used on shot #'
                                        +str(shot))
    else:
        # Change from manipulator units [mm] to m
        insertion /=1000
        geom_path = os.path.join(paths.ScintSuite, 'Data',
                             'Calibrations', 'FILD', 'AUG',
                             'GeometryDefaultParameters.txt')
        geom = f90nml.read(geom_path)[geomID]
        if insertion < geom['max_insertion'] \
            and insertion > geom['min_insertion']:
                R = geom['r_parking']-insertion*np.cos(geom['gamma']/180*np.pi)
                z = geom['z_parking']-insertion*np.sin(geom['gamma']/180*np.pi)
                position = {'R':R,
                            'z': z, 
                            'phi': geom['phi']}
        else:
            raise errors.NotValidInput('Insertion outside limits: ['+\
                                 str(geom['min_insertion'])+'; '+\
                                 str(geom['max_insertion'])+'] mm')
                    
    get_dist2sep(shot = shot, R = position['R'], z = position['z'],
                 t = t, diag = diag, plot_sep = plot_sep, 
                 plot_dist = plot_dist)

class FILD4_traject:
    """
    Class to handle FILD4 (magnetically diven FILD) trajectories.

    Javier Hidalgo - javih@us.es

    Public methods:
        - load_power_supply: Load the power supply output
        - reconstruct_traject: reconstruct FILD trajectory
        - get_dist2sep: get distance to separatrix
        - load_trajectory: Load FILD trajectory
    """

    def __init__(self, shot: int = None):
        """
        Initialise the class.

        :param  shot
        """
        self.shot = shot
        self.dat_ps = {}
        self.traject = {}

    def _linear_reg(x, y):
        n = np.size(x)
        m_x = np.mean(x)
        m_y = np.mean(y)

        # calculating cross-deviation and deviation about x
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x
        m = SS_xy / SS_xx
        n = m_y - m*m_x
        return (m, n)

    def load_power_supply(self, path_ps: str = '', smooth: bool = True,
                          win_length: int = 15, polyorder: int = 3):
        """
        Load the power supply output.

        This file has no headers. It was designed like this when FILD4 was
        first installed. They will be added in the future.

        :param  path_ps: path to the output of the power supply. If empty, the
        shot number will be used
        :param  smooth: smooth output
        :param  win_length: window of the Savgol filter
        :param  polyorder: polyorder of the Savgol filter
        """
        if path_ps == '':
            path_ps = os.path.join(paths.FILD4_trajectories,
                                   'output_power_supply', str(self.shot)[0:2],
                                   'FILD_MDRS_'+str(self.shot)+'.txt')
        print('path is ' + path_ps)
        data = np.loadtxt(path_ps, delimiter='\t')
        time_V_goal = data[:, 0]*1e-3  # s
        flag = int(np.argwhere(time_V_goal == np.max(time_V_goal))[0])
        time_V_goal = time_V_goal[0:flag]  # s
        V_goal = data[0:flag, 1]  # V

        time_I = data[:, 2]*1e-9  # s
        flag = int(np.argwhere(time_I == np.max(time_I))[0])
        time_I = time_I[0:flag]
        I = data[0:flag, 3]  # A
        time_V = data[0:flag, 4]*1e-9  # s
        V = data[0:flag, 5]  # V

        if smooth:
            V_goal = savgol(V_goal, win_length, polyorder)
            I = savgol(I, win_length, polyorder)
            V = savgol(V, win_length, polyorder)

        output = {'V_goal': V_goal,
                  'time_V_goal': time_V_goal,
                  'I': I,
                  'time_I': time_I,
                  'V': V,
                  'time_V': time_V}
        self.dat_ps = output
        return

    def reconstruct_traject(self, B: float = None, get_R: str = 'auto',
                            R: float = 13.5, R_fit_order: int = 0,
                            R_coef: tuple = (),
                            diag: str = 'EQH'):
        """
        Reconstruct FILD4 trajectory from the power supply output

        :param  B: toroidal magnetic field at the coil position. If None, it is
            calculated in the runtime
        :param  get_R: how is the resistance of the system calculated
            - 'auto': estimates in the code. Use R_fit_order to chose the
                order of the fit (0 or 1)
            - 'lineal': manually input the R time dependance
            - 'manual': just give a manual value
        :param  R_fit_order: order of the fit (1 or 0). Only for auto
        :param  R_coef: R = R_coef[0]*t+R_coef[1]. Only for lineal
        :param  R: R single value. Only for manual
        :param  diag: diagnostic for the equilibrium reconstruction
        """
        if not bool(self.dat_ps):
            raise NameError('Power supply not loaded. Run load_power_supply')

        max_insertion = 0.067  # m (FARO). Hardcoded
        # Get the magnetic field in the coil
        if B is None:
            equ = sf.EQU(self.shot, diag=diag)
            R_coil = params.FILD[3]['coil']['R_coil']
            Z_coil = params.FILD[3]['coil']['Z_coil']
            br, bz, bt = sf.rz2brzt(equ, r_in=R_coil, z_in=Z_coil,
                                    t_in=self.dat_ps['time_V'])
            B = abs(bt).squeeze()
        else:
            B = (abs(B)*np.ones(self.dat_ps['time_V'].shape)).squeeze()

        # Get the R value
        V = self.dat_ps['V']
        I = self.dat_ps['I']
        time_V = self.dat_ps['time_V']

        if get_R == 'auto':
            # Calculate R_fit from the R_output values that have reasonable
            # values -> 12 - 16 Ohm
            R_output = V/I
            # - Get the values of R during power supply operation
            index = abs(V) > 0.5
            R_output = R_output[index]
            time_R = time_V[index]
            index = (R_output < 16)*(R_output > 12)
            R_output = R_output[index]
            time_R = time_R[index]
            # Filter the values that are above the mean value
            flag = 0
            while flag == 0:
                R_mean = R_output.mean()
                time_R = time_R[abs((R_output-R_mean)) < 0.1*R_mean]
                R_output = R_output[abs((R_output-R_mean)) < 0.1*R_mean]
                if abs(R_mean-R_output.mean()) < 0.005*R_mean:
                    flag = 1
            if R_fit_order == 0:
                R_fit = R_output.mean()*np.ones(V.shape)
            elif R_fit_order == 1:
                coefs = self.linear_reg(time_R, R_output)
                R_fit = coefs[0]*time_V+coefs[1]
        elif get_R == 'lineal':
            R_fit = R_coef[0]*time_V+R_coef[1]
        elif get_R == 'manual':
            R_fit = R*np.ones(V.shape)

        # Calculate the vel and pos of the probe head
        l = params.FILD[3]['coil']['l']
        A = params.FILD[3]['coil']['A']
        N = params.FILD[3]['coil']['N']
        theta_parking = params.FILD[3]['coil']['theta_parking']

        # Initial conditions for the integration
        vel = np.empty(time_V.shape)
        pos = np.empty(time_V.shape)
        theta = np.empty(time_V.shape)
        pos[0] = 0
        theta[0] = np.arctan(pos[0]/l+np.tan(theta_parking))

        # v_simplified joins all non-theta dependant factor for readibility
        v_simplified = (V-I*R_fit)*l/(N*B*A)
        vel[0] = v_simplified[0]/np.cos(theta[0])**3

        # Trapezoidal integration
        for i in range(1, len(time_V)):
            vel[i] = v_simplified[i]/np.cos(theta[i-1])**3
            pos[i] = (vel[i]+vel[i-1])/2*(time_V[i]-time_V[i-1])+pos[i-1]
            if pos[i] > max_insertion:
                pos[i] = max_insertion
            elif pos[i] < 0:
                pos[i] = 0
            theta[i] = np.arctan(pos[i]/l + np.tan(theta_parking))

        self.traject = {'time': time_V,
                        'position': pos,
                        'velocity': vel,
                        'Resistance': R_fit}
        return

    def get_dist2sep(self, t: float = None, diag: str = 'EQH',
                     plot_sep: bool = True, plot_dist: bool = True):
        """
        Wrapper for the FILD4 case for the dist2sep routine.

        :param  t: time array. If None, entire shot
        :param  diag: diagnostic for equilibrium reconstruction
        :param  plot_sep: plot separatrix position
        :param  plot_dist: plot distance to the separatrix
        """
        if not bool(self.traject):
            raise NameError('Trajectory not loaded. Run reconstruct_traject')

        if t is None:
            t = self.traject['time']

        # Get FILD4 position
        insertion = np.interp(t, self.traject['time'],
                              self.traject['position'])
        R_pos = params.FILD[3]['coil']['R_parking']-insertion
        z_pos = params.FILD[3]['coil']['Z_parking']*np.ones(t.shape)
        self.dist2sep = get_dist2sep(shot=self.shot, R=R_pos, z=z_pos,
                                     t=t, diag=diag, plot_sep=plot_sep,
                                     plot_dist=False)

        # Adaptated plot_dist. In terms of insertion instead of absolute pos.
        if plot_dist:
            fig2, ax2 = plt.subplots(2)
            title = '#'+str(self.shot)
            fig2.suptitle(title)
            ax2[0].plot(t, insertion)
            ax2[0].set_xlabel('Time [s]')
            ax2[0].set_ylabel('Insertion [m]')
            ax2[1].plot(t, self.dist2sep)
            ax2[1].set_xlabel('Time [s]')
            ax2[1].set_ylabel('Dist2sep [m]')
            plt.show()

        return

    def load_trajectory(self, path_traject: str = '',
                        version: int = -1):
        """
        Load an already calculated trajectory.

        :param  path_traject: if unused. Check in the default folder
        :param  version: trajectory version to be loaded. -1 means last one
        """
        if path_traject == '':
            folder = os.path.join(paths.FILD4_trajectories,
                                  'reconstructed_trajectories',
                                  str(self.shot)[0:2])
            file_name = ''
            file_versions = []
            for file in os.listdir(folder):
                if file.startswith(str(self.shot)):
                    file_versions.append(file)
            if version == -1:
                file_name = file_versions[-1]
            else:
                for file in file_versions:
                    if file.endswith('_'+version+'.txt'):
                        file_name = file
            if file_name == '':
                raise NameError('Version not available')
            path_traject = os.path.join(folder, file_name)

        data = np.loadtxt(path_traject, skiprows=3, delimiter='\t')
        self.traject['time'] = data[:, 0]
        self.traject['position'] = data[:, 1]
        self.traject['velocity'] = data[:, 2]
        self.traject['Resistance'] = data[:, 3]

        return

    def save_trajectory(self, path_save: str = '', comment: str = ''):
        """
        Save the loaded trajectory.

        :param  path_save: If empty, save it in the default path. Takes into
            account previous versions.
        :param  comment
        """
        if not bool(self.traject):
            raise NameError('Trajectorynot loaded. Run reconstruct_traject')

        if path_save == '':
            folder = os.path.join(paths.FILD4_trajectories,
                                  'reconstructed_trajectories',
                                  str(self.shot)[0:2])
            if not os.path.isdir(folder):
                os.mkdir(folder)
            file_versions = []
            for file in os.listdir(folder):
                if file.startswith(str(self.shot)):
                    file_versions.append(file)
            filename = str(self.shot)+'_'+str(len(file_versions))+'.txt'
            path_save = os.path.join(folder, filename)

        print('Saving in '+path_save)
        with open(path_save, 'w') as f:
            f.write('#'+str(self.shot)+'   Date: '+str(date.today()))
            f.write('\nComment: ' + comment)
            f.write('\nTime [s]\tInsertion [m]\tVelocity [m/s]'
                    + '\tResistance [Ohm]')
            for i in range(len(self.traject['time'])):
                f.write('\n{:1.6}\t{:1.4}\t{:2.4}\t{:2.3}'.format(
                    self.traject['time'][i], self.traject['position'][i],
                    self.traject['velocity'][i],
                    self.traject['Resistance'][i]))
        return

    def plot_figs(self, traject: bool = True, power_supply: bool = True,
                  R_fit: bool = True):
        """
        :param  traject
        :param  power supply
        :param  R_fit
        """

        nplots = 0
        if power_supply:
            nplots += 2
        if traject:
            nplots += 2
        if R_fit:
            nplots += 1

        if nplots == 0:
            raise NameError('No plots chosen')

        id_plot = 0
        fig, ax = plt.subplots(nplots, sharex=True)
        fig.suptitle('#'+str(self.shot))
        if power_supply:
            ax[id_plot].plot(self.dat_ps['time_V'], self.dat_ps['V'], '.',
                             label='V power supply')
            ax[id_plot].plot(self.dat_ps['time_V_goal'], self.dat_ps['V_goal'],
                             '-r', label='V goal')
            ax[id_plot].set_ylabel('V [V]')
            ax[id_plot].grid(True)
            ax[id_plot].legend()
            id_plot += 1

            ax[id_plot].plot(self.dat_ps['time_I'], self.dat_ps['I'])
            ax[id_plot].set_ylabel('I [A]')
            ax[id_plot].grid(True)
            id_plot += 1

        if R_fit:
            ax[id_plot].plot(self.traject['time'], self.traject['Resistance'],
                             label = 'R_fit')
            ax[id_plot].plot(self.dat_ps['time_V'], 
                             self.dat_ps['V']/self.dat_ps['I'], label = 'V/I')
            ax[id_plot].grid(True)
            ax[id_plot].set_ylabel('R [Ohm]')
            ax[id_plot].set_ylim([10,18])
            id_plot += 1

        if traject:
            ax[id_plot].plot(self.traject['time'], self.traject['velocity'])
            ax[id_plot].grid(True)
            ax[id_plot].set_ylabel('vel [m/s]')
            id_plot += 1

            ax[id_plot].plot(self.traject['time'], self.traject['position'])
            ax[id_plot].set_ylabel('ins [m]')
            ax[id_plot].grid(True)
        ax[id_plot].set_xlabel('Time [s]')
        plt.show()
        return fig, ax
