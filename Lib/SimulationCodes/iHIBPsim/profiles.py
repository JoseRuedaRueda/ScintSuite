
"""
Object with profiles for iHIBPsim.

@todo The current approach is quite limited for any future expansion on the
profiles as it consists heavily on copy-pasting. It is proposed the following:
    a) Base class for a profile: contains a single profiles.
    b) Wrapper class for multiple profiles, just containing a collection.

@todo IMPORTANT: BE CAUTIOUS WHEN DEALING WITH X-POINT EQUILIBRIA. IT WILL
MAP THE REGIONS BELOW THE XPOINT (WHEN A LOWER XPOINT) AS AN ACTUAL PLASMA!!
"""
import numpy as np
import os
import Lib.LibData as dat
import warnings
import matplotlib.pyplot as plt

from scipy.interpolate import interpn, UnivariateSpline
from copy import copy
from Lib._Plotting import Gamma_II

class ihibpProfiles:
    """Class with iHIBPsim profiles."""

    def __init__(self):
        """
        Initializes a dummy object. Call the class methods:
        a) readFiles: to read from the files the appropriate profiles.
        b) readDB: to fetch the data from the database.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        self.ne = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64),
                   'dims': int(0),
                   'set': False
                   }

        self.Te = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64),
                   'dims': int(0),
                   'set': False
                   }

        self.ni = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64),
                   'dims': int(0),
                   'set': False
                  }
        self.nimp = {'R': np.array((0), dtype = np.float64),
                     'z': np.array((0), dtype = np.float64),
                     'f': np.array((0), dtype = np.float64),
                     'nPhi': np.array((1), dtype = np.int32),
                     'nTime': np.array((1), dtype = np.int32),
                     'Phimin': np.array((0.0), dtype = np.float64),
                     'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                     'Timemin': np.array((0.0), dtype = np.float64),
                     'Timemax': np.array((1.0), dtype = np.float64),
                     'dims': int(0),
                   'set': False
                  }

        self.Ti = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64),
                   'dims': int(0),
                   'set': False
                   }

        self.Zeff = {'R': np.array((0), dtype = np.float64),
                     'z': np.array((0), dtype = np.float64),
                     'f': np.array((0), dtype = np.float64),
                     'nPhi': np.array((1), dtype = np.int32),
                     'nTime': np.array((1), dtype = np.int32),
                     'Phimin': np.array((0.0), dtype = np.float64),
                     'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                     'Timemin': np.array((0.0), dtype = np.float64),
                     'Timemax': np.array((1.0), dtype = np.float64),
                     'dims': int(0),
                     'set': False
                     }

        self.from_shotfile = False  # If data is taken from DB.
        self.flag_ni_ne    = True   # Sets equality between electron density
                                    # and main ion density.
        self.flag_Zeff     = False  # Use Zeff to have the impurities.
        self.flag_Ti_Te    = True   # Sets equality between electron
                                    # temperature and main ion temperature.

        self.avg_charge = 5         # Average charge state of the impurities.

        # In case that we need to read the data from a DB, we will have 1D
        # profiles. We need variables to store 1D profiles and manipulate
        # them.
        self.grid_set_flag = False
        self.prof1D_flag = False
        self.prof1D      = {}

        # Internal variables to get the maps.
        self.equ_diag = 'EQI'
        self.equ_exp  = 'AUGD'
        self.interp_order = 3

    def setOneFluidModel(self):
        """
        This function allows to set the profiles of the class into the
        one-fluid model approximation, by equating the ion density to the
        electron density. Same for temperature.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.flag_ni_ne = True
        self.flag_Ti_Te = True

    def fromFiles(self, profName: str, fileName: str):
        """
        Reads the appropriate profile from the file and stores it into a
        variable inside the class. This will also create the appropriate
        interpolation routines.
        This routine has to be called for each one of the profiles that the
        class can store, i.e.:
            a) Electron density: 'ne'
            b) Electron temperature: 'Te'
            c) Main ion density: 'ni'
            d) Main ion temperature: 'Ti'
            e) Effective Z: 'Zeff'
            a) Impurity density: 'nimp'

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: Name of the input profile to read.
        :param  fileName: Path to the file containing the profile to read.
        """

        if profName not in self.__dict__.keys():
            raise ValueError('The profile %s is not in the class!'%profName)

        tmp = {}
        with open(fileName, 'rb') as fid:
            tmp['nR'] = np.fromfile(fid, 'uint32', 1)[0]
            tmp['nz'] = np.fromfile(fid, 'uint32', 1)[0]
            tmp['nPhi'] = np.fromfile(fid, 'uint32', 1)[0]
            tmp['nTime'] = np.fromfile(fid, 'uint32', 1)[0]

            tmp['Rmin'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Rmax'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Zmin'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Zmax'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Phimin'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Phimax'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Timemin'] = np.fromfile(fid, 'float64', 1)[0]
            tmp['Timemax'] = np.fromfile(fid, 'float64', 1)[0]

            size2read = tmp['nR']   * tmp['nz'] *\
                        tmp['nPhi'] * tmp['nTime']

            data = np.fromfile(fid, 'float64', count=size2read)

            # Generating the grids.
            grr  = np.linspace(tmp['Rmin'], tmp['Rmax'], num=tmp['nR'])
            gzz  = np.linspace(tmp['Zmin'], tmp['Zmax'], num=tmp['nz'])
            gphi = np.linspace(tmp['Phimin'], tmp['Phimax'], num=tmp['nPhi'])
            gtt  = np.linspace(tmp['Timemin'], tmp['Timemax'],
                               num=tmp['nTime'])

            tmp['R'] = grr
            tmp['Z'] = gzz
            tmp['Phi'] = gphi
            tmp['Time'] = gtt

            grid = np.array((tmp['nR'], tmp['nPhi'], tmp['nz'], tmp['nTime']))

            tmp['f'] = data.reshape(grid, order='F').squeeze()
            tmp['set'] = True # Variable loaded!

            # Dividing by dimensionality.
            if tmp['nPhi'] == 1 and tmp['nTime'] == 1:
                tmp['dims'] = 2
                tmp['interp'] = lambda r, z, phi, time: \
                                interpn((grr, gzz), tmp['f'],
                                        (r.flatten(), z.flatten()))

            elif tmp['nPhi'] != 1 and tmp['nTime'] == 1:
                tmp['dims'] = 3
                tmp['interp'] = lambda r, z, phi, time: \
                                interpn((grr, gzz, gphi), tmp['f'],
                                        (r.flatten(), phi.flatten(), \
                                         z.flatten()))
            else:
                tmp['dims'] = 4
                tmp['interp'] = lambda r, z, phi, time: \
                                interpn((grr, gzz, gphi, gtt), tmp['f'],
                                        (r.flatten(), phi.flatten(), \
                                         z.flatten(), time.flatten()))

            # --- Saving the data to the corresponding field:
            profName = profName.lower()
            if profName == 'ne':
                self.ne = tmp
            elif profName == 'Te':
                self.te = tmp
            elif profName == 'ni':
                self.flag_ni_ne = False
                self.ni = tmp
            elif profName == 'ti':
                self.flag_Ti_Te = False
                self.Ti = tmp

    def toFile(self, profName: str, filename: str, overwrite: bool=True):
        """"
        Stores the corresponding profile (profName) to the filename such that
        i-HIBPsim can read it directly.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name of the profile to store into the file.
        :param  filename: name of the output file.
        """

        if profName not in self.__dict__.keys():
            raise ValueError('The profile %s is not in the class!'%profName)

        tmp = self.__dict__[profName]

        if not tmp['set']:
            if profName in self.prof1D:
                self.__1d_to_nDprofiles((profName,))
            else:
                raise Exception('The profile %s is not initiated!!'%profName)

        if (not overwrite) and os.path.isfile(filename):
            raise FileExistsError('The file %s already exists!')

        with open(filename, 'wb') as fid:
            # Writing the header.
            gridsize = (tmp['nR'], tmp['nz'], tmp['nPhi'], tmp['nTime'])
            np.array(gridsize, dtype='uint32').tofile(fid)

            gridlimits = (tmp['Rmin'], tmp['Rmax'], tmp['Zmin'], tmp['Zmax'],
                          tmp['Phimin'], tmp['Phimax'], tmp['Timemin'],
                          tmp['Timemax'])

            np.array(gridlimits, dtype='float64').tofile(fid)

            # Writing the profile data.
            np.array(tmp['f'], dtype='float64').ravel(order='F').tofile(fid)

        return

    def getfromDB(self, shotnumber: int, profName: str, diag: str=None,
                  exp: str='AUGD', time: float=None, edition: int=0,
                  flag_avg: bool = False, ii_avg: int = 8):
        """
        Routines to read the profiles from the database and store them into the
        class.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  shotnumber: shotnumber to retrieve the profiles.
        :param  profName: name of the profile to read from the data base.
        :param  diag: diagnostic to read the data from. If None, the default is
        chosen accordingly to the input profName.
        :param  exp: experiment where the shotfile is saved. By default, 'AUGD'
        """

        if profName.lower() not in ('ne', 'ni', 'te', 'ti'):
            raise NotImplementedError('Profile %s not yet implemented.'%profName)

        if diag is None:
            STANDARD_DIAGS = { 'ne': 'IDA',
                               'te': 'IDA',
                               'ni': 'IDI',
                               'ti': 'IDI',
                             }
            diag = STANDARD_DIAGS[profName.lower()]

        if profName.lower() == 'ne':
            data = dat.get_ne(shotnumber=shotnumber, exp=exp, diag=diag,
                              edition=edition,time=time, flag_avg = flag_avg,
                              ii_avg = ii_avg)

            self.prof1D['ne'] = { 'shotnumber': shotnumber,
                                  'diag': diag,
                                  'edition': edition,
                                  'time': data['time'],
                                  'rhop': data['rhop'],
                                  'data': data['data'],
                                  'flag_perturb': False,
                                  'func': None,
                                  'kwargs': {}
                                }

            if not self.grid_set_flag:
                warnings.warn('Profile is loaded, but grid is not defined!')

        elif profName.lower() == 'te':
            data = dat.get_Te(shotnumber=shotnumber, exp=exp, diag=diag,
                              edition=edition,time=time, flag_avg=flag_avg,
                              ii_avg=ii_avg )

            self.prof1D['Te'] = { 'shotnumber': shotnumber,
                                  'diag': diag,
                                  'edition': edition,
                                  'time': data['time'],
                                  'rhop': data['rhop'],
                                  'data': data['data'],
                                  'flag_perturb': False,
                                  'func': None,
                                  'kwargs': {}
                                }

            if not self.grid_set_flag:
                warnings.warn('Profile is loaded, but grid is not defined!')

        elif profName.lower() == 'ti':
            data = dat.get_Ti(shotnumber=shotnumber, exp=exp, diag=diag,
                              edition=edition,time=time)

            self.prof1D['Ti'] = { 'shotnumber': shotnumber,
                                  'diag': diag,
                                  'edition': edition,
                                  'time': data['time'],
                                  'rhop': data['rhop'],
                                  'data': data['data'],
                                  'flag_perturb': False,
                                  'func': None,
                                  'kwargs': {}
                                }

            if not self.grid_set_flag:
                warnings.warn('Profile is loaded, but grid is not defined!')

        elif profName.lower() == 'ni':
            raise NotImplementedError('Ion density reading not implemented!')
            # data = dat.get_Ti(shotnumber=shotnumber, exp=exp, diag=diag,
            #                   edition=edition,time=time)

            self.prof1D['ni'] = { 'shotnumber': shotnumber,
                                  'diag': diag,
                                  'edition': edition,
                                  # 'time': data['time'],
                                  # 'rhop': data['rhop'],
                                  # 'data': data['data'],
                                  # 'flag_perturb': False,
                                  # 'func': None,
                                  # 'kwargs': {}
                                }

            if not self.grid_set_flag:
                warnings.warn('Profile is loaded, but grid is not defined!')

    def set_perturbation(self, profName: str, f: callable, **kwargs):
        """
        This routine will incorporate to the 1D profiles the capabilities to
        include variations in the profile as a function of the rhopol variable.

        The function f should be a callable function that must accept at least
        two inputs: (rhop, ne_old, **kwargs) and keyword arguments and must
        return a new profile.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name of the profile to modify.
        :param  f: callable function to modify the profile.
        :param  kwargs: keyword arguments to pass down to the function f.
        """

        if profName not in self.prof1D:
            raise ValueError('The profile %s has not been yet loaded!'%profName)

        self.prof1D[profName]['flag_perturb'] = True
        self.prof1D[profName]['func'] = f
        self.prof1D[profName]['kwargs'] = kwargs

        self.__dict__[profName]['set'] = False

    def change_pert_params(self, profName: str, **kwargs):
        """
        This routine will incorporate to the 1D profiles the capabilities to
        include variations in the profile as a function of the rhopol variable.

        The function f should be a callable function that must accept at least
        two inputs: (rhop, ne_old, **kwargs) and keyword arguments and must
        return a new profile.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name of the profile to modify.
        :param  f: callable function to modify the profile.
        :param  kwargs: keyword arguments to pass down to the function f.
        """

        if profName not in self.prof1D:
            raise ValueError('The profile %s has not been yet loaded!'%profName)

        if not self.prof1D[profName]['flag_perturb']:
            raise ValueError('The peturbation is not set for profile %s'%profName)

        self.prof1D[profName]['flag_perturb'] = True
        self.prof1D[profName]['kwargs'] = kwargs

        self.__dict__[profName]['set'] = False

    def reset(self, profName: str=None):
        """
        Changes the value of the set status of a given profile name.

        If the profile name is not provided, then all the variables will be set
        to False.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name(s) of the profiles to unset.
        """

        if profName is None:
            profName = ('ne', 'Te', 'ni', 'Ti')
        else:
            if isinstance(profName, str):
                profName = (profName,)

            for ii in profName:
                if ii not in ('ne', 'Te', 'ni', 'Ti'):
                    raise ValueError('Profile %s not supported'%ii)

        profName = np.array((profName), dtype=str)

        # Loop of the profiles.
        for ii in profName:
            self.__dict__[ii]['set'] = False


    def set_grid(self, rmin: float, rmax: float, zmin: float, zmax: float,
                 phimin: float=0.0, phimax:float=2.0*np.pi,
                 tmin: float=None, tmax: float=None, nR: int=128,
                 nz: int=256, nphi: int=1, ntime: int=1):
        """
        Sets the internal grid to transform the 1D rhopol into 2D+time such it
        can be readily written into files.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  rmin: minimum major radius.
        :param  rmax: maximum major radius.
        :param  zmin: minimum vertical position.
        :param  zmax: maximum vertical position.
        :param  phimin: minimum toroidal position.
        :param  phimax: maximum toroidal position.
        :param  timemin: minimum time point to simulate. If None, then the time
        dimension is disregarded.
        :param  timemax: maximum time point to simulate. If None, then the time
        dimension is disregarded.
        """

        self.grids = { 'rmin': rmin,
                       'rmax': rmax,
                       'nR':   nR,
                       'R': np.linspace(rmin, rmax, nR),

                       'zmin': zmin,
                       'zmax': zmax,
                       'nz':   nz,
                       'Z': np.linspace(zmin, zmax, nz),

                       'dims': 2
                     }

        self.grids['Rgrid'], self.grids['Zgrid'] = np.meshgrid(self.grids['R'],
                                                               self.grids['Z'])

        self.grids

        if (phimin == phimax) or (nphi == 1):
            self.grids['phimin'] = 0.0
            self.grids['phimax'] = 2.0*np.pi
            self.grids['nPhi']   = 1
            self.grids['Phi']    = np.array((0.0,), dtype='float64')
        else:
            self.grids['phimin'] = phimin
            self.grids['phimax'] = phimax
            self.grids['nPhi']   = nphi
            self.grids['Phi']    = np.linspace(phimin, phimax, nphi)
            self.grids['dims']   = 3

        if (tmin == tmax) or (ntime == 1):
            self.grids['timemin'] = 0.0
            self.grids['timemax'] = 1.0
            self.grids['nTime']   = 1
            self.grids['Time']    = np.array((0.0,), dtype='float64')
        elif (self.grids['ndims'] != 3):
            self.grids['timemin'] = tmin
            self.grids['timemax'] = tmax
            self.grids['nTime']   = ntime
            self.grids['Time']    = np.linspace(tmin, tmax, ntime)
            self.grids['dims']    = 4
        else:
            self.grids['timemin'] = tmin
            self.grids['timemax'] = tmax
            self.grids['nTime']   = ntime
            self.grids['Time']    = np.linspace(tmin, tmax, ntime)
            self.grids['dims']    = 4

        self.grid_set_flag = True

    def __1d_to_grid(self, data: dict):
        """
        Internal helper routine to tranform from the 1d rhopol into (R,z).

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  data: dictionary: dictionary with the data of the profiles in 1D.
        """

        # To make the transformation from rhopol 2 Rz we need to get the
        # psipol corresponding to the (R,z) at each time point.

        rhopol = dat.get_rho(shot=data['shotnumber'], diag=self.equ_diag,
                             exp=self.equ_exp, coord_out='rho_pol',
                             Rin=self.grids['Rgrid'].T.flatten(),
                             zin=self.grids['Zgrid'].T.flatten(),
                             time=data['time'])

        interpolator = UnivariateSpline(data['rhop'], data['data'],
                                        k=self.interp_order, s=0,
                                        ext='zeros')

        rhopol = rhopol.reshape(self.grids['nR'],self.grids['nz'],-1)
        print(rhopol.shape)

        return interpolator(rhopol)

    def __1d_to_nDprofiles(self, names: str=None):
        """
        Transforms the 1D profiles into nD profiles and store them into the
        main variables of the class.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        if names is None:
            names = np.array(self.prof1D.keys(), dtype=str)
        else:
            if isinstance(names, str):
                names = (names,)

            names = np.array(names, dtype=str)

        for ii in names:
            data_tmp = copy(self.prof1D[ii])
            if data_tmp['flag_perturb']:
                data_new = data_tmp['func'](data_tmp['rhop'],
                                            data_tmp['data'],
                                             **data_tmp['kwargs'])

                data_tmp['data'] = data_new


            a = self.__1d_to_grid(data_tmp).squeeze()
            tmp = self.__dict__[ii]

            if self.grids['dims'] == 3: # NO time dependence.
                a = np.tile(a, (self.grids['nPhi'], 1, 1))
                a = np.moveaxis(a, source=0, destination=-1)
            elif self.grids['dims'] == 4: # time dependence.
                a = np.tile(a, (1,1,1,self.grids['nPhi']))
                a = np.moveaxis(a, source=0, destination=-2)

            tmp['Rmin'] = self.grids['rmin']
            tmp['Rmax'] = self.grids['rmax']
            tmp['Zmin'] = self.grids['zmin']
            tmp['Zmax'] = self.grids['zmax']
            tmp['Phimin'] = self.grids['phimin']
            tmp['Phimax'] = self.grids['phimax']
            tmp['Timemin'] = self.grids['timemin']
            tmp['Timemax'] = self.grids['timemax']
            tmp['ndims']   = self.grids['dims']
            tmp['R'] = self.grids['R']
            tmp['Z'] = self.grids['Z']
            tmp['phi'] = self.grids['Phi']
            tmp['Time'] = self.grids['Time']
            tmp['nR'] = self.grids['nR']
            tmp['nz'] = self.grids['nz']
            tmp['nPhi'] = self.grids['nPhi']
            tmp['nTime'] = self.grids['nTime']
            tmp['f'] = a
            tmp['set'] = True

    def plot1d(self, profName:str,  ax, fig, timeSlice: int=0,
               line_params: dict={}, setLabels: bool=False):
        """
        Plotting function for the 1D profiles as a function of the rhopol
        variable.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name of the profile to be plotted.
        :param  view: how to represent the profiles. It can either be 1D or
        2D, but if 1D is established, the profiles should have been read
        from the database.
        :param  ax: axis to plot the data. If None, new ones will be created.
        :param  fig: figure where the axis lie. If None, current one will be
        retrieved.
        :param  timeSlice: time slice to plot, in case the profile is time
        dependent. If NOne, the first time point is used.
        :param  line_params: dictionary to be passed down to plt.plot.
        """

        # Checking if the name is in the 1D data.
        if (profName not in self.prof1D) and (profName in self.__dict__.keys()):
            raise NotImplementedError('The name of the profile'+\
                                      ' %s is not in the 1D list, '%profName+\
                                      'but on the 2D.')

        if self.prof1D[profName]['flag_perturb']:
            print('hola')
            toplot = self.prof1D[profName]['func'](self.prof1D[profName]['rhop'],
                                                   self.prof1D[profName]['data'],
                                                   **self.prof1D[profName]['kwargs'])
        else:
            toplot = self.prof1D[profName]['data']

        # If the profile is time-dependent, data should have a second dimension
        # corresponding to the time axis.
        if toplot.ndim == 2:
            toplot = toplot[:, timeSlice]

        im = ax.plot(self.prof1D[profName]['rhop'], toplot, **line_params)

        if setLabels:
            ax.set_xlabel('$\\rho_{pol}$')

            ylabel = { 'ne': 'Electron density [$m^{-3}$]',
                       'te': 'Electron temperature [eV]',
                       'ni': 'Main ion density [$m^{-3}$]',
                       'ti': 'Main ion temperature [eV]',
                     }[profName.lower()]
            ax.set_ylabel(ylabel)

        return im, ax

    def plot2d(self, profName: str, ax, fig,phiSlice: int=0, timeSlice: int=0,
               surf_params: dict={}, setLabels: bool=False):
        """
        Plotting function for the 1D profiles as a function of the rhopol
        variable.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name of the profile to be plotted.
        :param  view: how to represent the profiles. It can either be 1D or
        2D, but if 1D is established, the profiles should have been read
        from the database.
        :param  ax: axis to plot the data. If None, new ones will be created.
        :param  fig: figure where the axis lie. If None, current one will be
        retrieved.
        :param  timeSlice: time slice to plot, in case the profile is time
        dependent. If NOne, the first time point is used.
        :param  line_params: dictionary to be passed down to plt.plot.
        """

        if (profName not in self.__dict__.keys()) and\
           (profName not in self.prof1D):
            raise ValueError('The profile is not in the database.')

        tmp = self.__dict__[profName]

        # If the variable has not yet been converted into (R, z) we do it
        # with the internal routine.
        if not tmp['set'] and profName in self.prof1D:
            self.__1d_to_nDprofiles((profName,))
            tmp = self.__dict__[profName]

        if 'cmap' not in surf_params:
            surf_params['cmap'] = Gamma_II()

        im = ax.contourf(tmp['R'], tmp['Z'], tmp['f'].T, **surf_params)

        if setLabels:
            ax.set_xlabel('Major radius [m]')
            ax.set_ylabel('Vertical Z [m]')

            cbar=fig.colorbar(mappable=im, ax=ax)

            clabel = { 'ne': 'Electron density [$m^{-3}$]',
                       'te': 'Electron temperature [eV]',
                       'ni': 'Main ion density [$m^{-3}$]',
                       'ti': 'Main ion temperature [eV]',
                     }[profName.lower()]
            cbar.ax.set_ylabel(clabel)
        else:
            cbar=None
        return im, cbar

    def plot(self, profName: str, view: str='2D', ax=None, fig=None,
             phiSlice: int=0, timeSlice: int=0, **kwargs):
        """
        Wrapper to the plotting routines of the profiles. This routine
        can plot 1D or 2D profiles if needed.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  profName: name of the profile to be plotted.
        :param  view: how to represent the profiles. It can either be 1D or
        2D, but if 1D is established, the profiles should have been read
        from the database.
        :param  ax: axis to plot the data. If None, new ones will be created.
        :param  fig: figure where the axis lie. If None, current one will be
        retrieved.
        :param  phiSlice: in case the profile is 3D (non-axisymmetric), then
        the phi slice has to be chosen beforehand. If None, phi=0 is taken.
        :param  timeSlice: time slice to plot, in case the profile is time
        dependent. If NOne, the first time point is used.
        :param  kwargs: keyword arguments to be passed down to the actual
        plotting functions.
        """

        axis_needed = False
        if ax is None:
            fig, ax = plt.subplots(1)
            axis_needed = False

        if view == '1D':
            return self.plot1d(profName, ax, fig=plt.gcf(),
                               timeSlice=timeSlice, setLabels=~axis_needed,
                               line_params=kwargs)

        elif view == '2D':
            return self.plot2d(profName, ax, fig=plt.gcf(), phiSlice=phiSlice,
                               timeSlice=timeSlice, setLabels=~axis_needed,
                               surf_params=kwargs)
