"""Object with profiles for iHIBPsim."""
import numpy as np
from scipy.interpolate import interpn


class ihibpProfiles:
    """Class with iHIBPsim profiles."""

    def __init__(self):
        """
        Initializes a dummy object. Call the class methods:
        a) readFiles: to read from the files the appropriate profiles.
        b) readDB: to fetch the data from the database.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False
        self.ne = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64)
                   }

        self.Te = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64)
                   }

        self.ni = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64)
                  }
        self.nimp = {'R': np.array((0), dtype = np.float64),
                     'z': np.array((0), dtype = np.float64),
                     'f': np.array((0), dtype = np.float64),
                     'nPhi': np.array((1), dtype = np.int32),
                     'nTime': np.array((1), dtype = np.int32),
                     'Phimin': np.array((0.0), dtype = np.float64),
                     'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                     'Timemin': np.array((0.0), dtype = np.float64),
                     'Timemax': np.array((1.0), dtype = np.float64)
                  }

        self.Ti = {'R': np.array((0), dtype = np.float64),
                   'z': np.array((0), dtype = np.float64),
                   'f': np.array((0), dtype = np.float64),
                   'nPhi': np.array((1), dtype = np.int32),
                   'nTime': np.array((1), dtype = np.int32),
                   'Phimin': np.array((0.0), dtype = np.float64),
                   'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                   'Timemin': np.array((0.0), dtype = np.float64),
                   'Timemax': np.array((1.0), dtype = np.float64)
                   }

        self.Zeff = {'R': np.array((0), dtype = np.float64),
                     'z': np.array((0), dtype = np.float64),
                     'f': np.array((0), dtype = np.float64),
                     'nPhi': np.array((1), dtype = np.int32),
                     'nTime': np.array((1), dtype = np.int32),
                     'Phimin': np.array((0.0), dtype = np.float64),
                     'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                     'Timemin': np.array((0.0), dtype = np.float64),
                     'Timemax': np.array((1.0), dtype = np.float64)
                     }

        self.from_shotfile = False  # If data is taken from DB.
        self.flag_ni_ne    = True   # Sets equality between electron density
                                    # and main ion density.
        self.flag_Zeff     = False  # Use Zeff to have the impurities.
        self.flag_Ti_Te    = True   # Sets equality between electron
                                    # temperature and main ion temperature.

        self.avg_charge = 5         # Average charge state of the impurities.

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

        @param profName: Name of the input profile to read.
        @param fileName: Path to the file containing the profile to read.
        """

        tmp = {}
        with open(profName, 'rb') as fid:
            tmp['nR'] = np.fromfile(fid, 'uint32', 1)
            tmp['nZ'] = np.fromfile(fid, 'uint32', 1)
            tmp['nPhi'] = np.fromfile(fid, 'uint32', 1)
            tmp['nTime'] = np.fromfile(fid, 'uint32', 1)

            tmp['Rmin'] = np.fromfile(fid, 'float64', 1)
            tmp['Rmax'] = np.fromfile(fid, 'float64', 1)
            tmp['Zmin'] = np.fromfile(fid, 'float64', 1)
            tmp['Zmax'] = np.fromfile(fid, 'float64', 1)
            tmp['Phimin'] = np.fromfile(fid, 'float64', 1)
            tmp['Phimax'] = np.fromfile(fid, 'float64', 1)
            tmp['Timemin'] = np.fromfile(fid, 'float64', 1)
            tmp['Timemax'] = np.fromfile(fid, 'float64', 1)

            size2read = tmp['nR']   * tmp['nZ'] *\
                        tmp['nPhi'] * tmp['nTime']

            data = np.fromfile(fid, 'float64', count=size2read[0])

            # Generating the grids.
            grr  = np.linspace(tmp['Rmin'], tmp['Rmax'], num=tmp['nR'])
            gzz  = np.linspace(tmp['Zmin'], tmp['Zmax'], num=tmp['nZ'])
            gphi = np.linspace(tmp['Phimin'], tmp['Phimax'], num=tmp['nPhi'])
            gtt  = np.linspace(tmp['Timemin'], tmp['Timemax'],
                               num=tmp['nTime'])

            grid = np.array((tmp['nR'][0], tmp['nPhi'][0], tmp['nZ'][0],
                             tmp['nTime']))

            tmp['f'] = data.reshape(grid).squeeze()

            # --- Dividing by dimensionality.
            if tmp['nPhi'][0] == 1 and tmp['nTime'][0] == 1:
                tmp['dims'] = 2
                tmp['interp'] = lambda r, z, phi, time: \
                                interpn((grr, gzz), tmp['f'],
                                        (r.flatten(), z.flatten()))

            elif tmp['nPhi'][0] != 1 and tmp['nTime'][0] == 1:
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
            elif profName == 'te':
                self.te = tmp
            elif profName == 'ni':
                self.flag_ni_ne = False
                self.ni = tmp
            elif profName == 'Ti':
                self.flag_Ti_Te = False
                self.Ti = tmp
