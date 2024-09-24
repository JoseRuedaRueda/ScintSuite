"""
Retrieves EFIT equilibrium data from an MDS plus server

Jose Rueda Rueda: jruedaru@uci.edu

Adapted form the FILDcam program written by Kenny and
Xiaodi Du INPASIM
"""

from tkinter import N
import MDSplus
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import logging
logger = logging.getLogger('ScintSuite.MDSplus')
class EFIT:
    """
    Class for aqcuiring and storing MDS+ data for a given shot & efit tree.
    The structure stores data for all available times, but gives easy access
    to one data at a single time.
    
    This work for D3D COCOS, I won't bet on it working for other machines
    """
    def __init__(self, efit: str='efit02', 
                 shot=None, server: str='atlas.gat.com',
                 precalculateInterpolators: bool=True):
        """
        Class to get efit data from MDSplus server

        :param efit: name of the efit tree to load
        :param shot: shot number to load
        :param server: server from where retreive data
        :param precalculateInterpolators: If true calculate f interpolators
            These are used for the B field calculation
        """
        self._data = {}
        self.efit = (None, None)
        self.shot = None
        self.server = server
        if efit is not None and shot is not None:
            self.acquire_efit(efit, shot)
        if precalculateInterpolators:
            self.calculateFinterpolators()
    def __getitem__(self, key):
        return self._data[key]
    def __setitem__(self, key, value):
        self._data[key] = value

    def acquire_efit(self, efit, shot):
        """
        Retrieves MDS+ data required for FILD calculations.

        :param efit(str): string name of efit tree to pull data from
        :param shot(int): shot number to work with

        :return: None, just put things inside
        """
        if (efit, shot) == self.efit:
            # We already have this data, do nothing
            return

        try:
            c = MDSplus.Connection(self.server)
            c.openTree(efit, shot)
        except MDSplus.MdsException:
            print('This MDS+ efit tree does not exist')
            return

        self.efit = (efit, shot)

        self['GTIME'] = c.get('\GTIME').data()      # Times for each data point
        self['nT'] = len(self['GTIME'])
        self['R'] = c.get('\R').data()              # R-coordinate values
        self['nR'] = len(self['R'])
        self['Z'] = c.get('\Z').data()              # Z-coordinate values
        self['nZ'] = len(self['Z'])
        self['PSIRZ'] = c.get('\PSIRZ').data()      # Polidal flux (rz grid)
        self['CPASMA'] = c.get('\CPASMA').data()    # Plasma current
        self['SSIBRY'] = c.get('\SSIBRY').data()    # Poloidal Flux (boundary)
        self['SSIMAG'] = c.get('\SSIMAG').data()    # Poloidal flux (Mag axis)
        self['BCENTR'] = c.get('\BCENTR').data()    # Vacuum torr B @ center
        self['RZERO'] = c.get('\RZERO').data()      # R at center of LCFS
        self['FPOL'] = c.get('\FPOL').data()        # Poloidal current func
                                                        # F = R * Bt?
        self['LIM'] = c.get('\LIM').data()          # R,Z of limiter
        self['RMAXIS'] = c.get('\RMAXIS').data()    # R-coord of mag axis
        self['ZMAXIS'] = c.get('\ZMAXIS').data()    # Z-coord of mag axis
        self['BDRY'] = c.get('\BDRY').data()        # R,Z of boundary
        
        self['MH'] = len(self['Z'])                 # Number Z points
        self['MW'] = len(self['R'])                 # Number R points
        self['ZMID'] = (self['Z'][0]+
                        self['Z'][-1])/2.0          # Center of Z grid (zero)

        # Get the rhopol and q profile
        ntimes, nrho = self['FPOL'].shape
        psi_arr = np.zeros((ntimes, nrho))
        rho = np.zeros((ntimes, nrho))

        for i in range(ntimes):
            psi_arr[i, :] = np.linspace(self['SSIMAG'][i], self['SSIBRY'][i], nrho)
            rho[i, :] = np.sqrt(np.abs(psi_arr[i, :]-self['SSIMAG'][i])/np.abs(self['SSIBRY'][i]-self['SSIMAG'][i]))
        self['RHOPOL'] = rho.mean(axis=0) # By construction, rhopol is the same for all times
        # self['RHOPOL'] = rho
        self['Q'] = c.get('\QPSI').data()
        self['q'] = None
        c.closeTree(efit, shot)
        logger.debug('nr: %i'%self['nR'])
        logger.debug('nz: %i'%self['nR'])
        logger.debug('psi shape: (%i, %i, %i)'%self['PSIRZ'].shape)

        
    def closest_time(self, time):
        """
        Sets data retrieval to give output for the time closest to given.

        Parameters:
            time(float): time to get data for.

        Returns:
            index(int): index of time closest to given
        """
        return (np.abs(self._data['GTIME'] - time)).argmin()
    
    def calculateFinterpolators(self,):
        """
        Calculate the interpolators of F for the B calculation
        """
        # first deravitive of the psi in RZ direction
        dpsi_dr = np.zeros([self['nT'], self['nR'],self['nZ']])
        dpsi_dz = np.zeros([self['nT'], self['nR'],self['nZ']])
        r = self['R']
        z = self['Z']
        logger.debug(self['PSIRZ'].shape)
        # Swap axis such that the first spatial axis is the radius one
        psirz = np.swapaxes(self['PSIRZ'], 1,2)
        logger.debug(psirz.shape)
        for it in range(self['nT']):
            for i in range(1, self['nR']-1):
                for j in range(1, self['nZ']-1):
                    dpsi_dr[it,i,j] =\
                        (psirz[it, i+1,j]-psirz[it, i-1,j])/(r[i+1]-r[i-1])
                    dpsi_dz[it,i,j] =\
                        (psirz[it, i,j+1]-psirz[it, i,j-1])/(z[j+1]-z[j-1])
        # interpolate f
        logger.debug('Getting Psi inteprolartor')
        psirz = np.array((psirz-self['SSIMAG'][:, None, None])/\
                         (self['SSIBRY'][:, None, None]-self['SSIMAG'][:, None, None]),)
        fpsi = RegularGridInterpolator((self['GTIME'], r, z), psirz, 
                                       method='cubic', fill_value=0.0,
                                       bounds_error=False)
        logger.debug('Getting dr interpolator')
        fr = RegularGridInterpolator((self['GTIME'], r, z), dpsi_dr, 
                                     method='cubic', fill_value=0.0,
                                     bounds_error=False)
        logger.debug('Getting dz interpolator')
        fz = RegularGridInterpolator((self['GTIME'], r, z), dpsi_dz, 
                                     method='cubic', fill_value=0.0,
                                     bounds_error=False)

        self['fpsi'] = fpsi
        self['fr'] = fr
        self['fz'] = fz
        self['rc'] = interp1d(self['GTIME'], self['RZERO'], bounds_error=False,
                              fill_value=0.0)
        self['bc'] = interp1d(self['GTIME'], self['BCENTR'], bounds_error=False,
                              fill_value=0.0)

        return
    
    def calculate_q_interpolator(self):
        """
        Calculate the q interpolator
        """
        q = RegularGridInterpolator((self['GTIME'], self['RHOPOL']), self['Q'],
                                    method='cubic', fill_value=0.0, 
                                    bounds_error=False,)
        self['q'] = q
        return
    
    def Bfield(self, time, r, z):
        """
        Calculate the B field at a given time and position

        :param time: time to calculate the B field
        :param r: R position to calculate the B field
        :param z: Z position to calculate the B field

        :return: B field at the given position and time
        """
        # Get the interpolators and factors from the object
        fpsi = self['fpsi']
        fr = self['fr']
        fz = self['fz']
        rc = self['rc']
        bc = self['bc']
        cpasma = self['CPASMA'] # equ in res in function bfield.prepare
        currentsign = np.sign(cpasma)[0]
        # local magnetic field strength
        l_dpsir = fr((time, r, z))
        l_dpsiz = fz((time, r, z))
        br = currentsign*-1.0*-(1/r)*l_dpsiz
        bz = currentsign*-1.0* (1/r)*l_dpsir
        bt = np.asarray(rc(time)*bc(time)/r)
        return br, bz, bt

    def q(self, time, rhopol):
        """
        Calculate the q profile at a given time and rhopol

        :param time: time to calculate the q profile
        :param rhopol: rhopol to calculate the q profile

        :return: q profile at the given time and rhopol
        """
        if self['q'] is None:
            self.calculate_q_interpolator()
        
        return self['q']((time, rhopol))

