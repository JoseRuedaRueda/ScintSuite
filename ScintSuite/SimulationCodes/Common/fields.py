"""Routines to read and write the fields for PSFT simulation codes"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import ScintSuite.LibData as ssdat
import ScintSuite._Plotting as ssplt
import ScintSuite.errors as errors
import math
from ScintSuite._Machine import machine
import os
import netCDF4 as nc


class fields:
    """
    General class for iHIBPsim, SINPA and the new FILDSIM codes.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    feat Jose Rueda - jrrueda@us.es
    """

    def __init__(self):
        """
        Initialize a dummy object.

        Call the class methods:
            a) readFiles: to read from the files the EM fields.
            b) readBfromAUG: to fetch the fields from the AUG database.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :Example:
        >>> fields = fields()
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False
        self.psipol_type = None
        self.Bfield = {
            'R': np.array((0), dtype=np.float64),
            'nR': np.array((1), dtype=np.int32),
            'z': np.array((0), dtype=np.float64),
            'nz': np.array((1), dtype=np.int32),
            'fr': np.array((0), dtype=np.float64),
            'fz': np.array((0), dtype=np.float64),
            'ft': np.array((0), dtype=np.float64),
            'nPhi': np.array((1), dtype=np.int32),
            'nTime': np.array((1), dtype=np.int32),
            'Phimin': np.array((0.0), dtype=np.float64),
            'Phimax': np.array((2.0*np.pi), dtype=np.float64),
            'Timemin': np.array((0.0), dtype=np.float64),
            'Timemax': np.array((1.0), dtype=np.float64)
        }

        self.Efield = {
            'R': np.array((0), dtype=np.float64),
            'nR': np.array((1), dtype=np.int32),
            'z': np.array((0), dtype=np.float64),
            'nz': np.array((1), dtype=np.int32),
            'fr': np.array((0), dtype=np.float64),
            'fz': np.array((0), dtype=np.float64),
            'ft': np.array((0), dtype=np.float64),
            'nPhi': np.array((1), dtype=np.int32),
            'nTime': np.array((1), dtype=np.int32),
            'Phimin': np.array((0.0), dtype=np.float64),
            'Phimax': np.array((2.0*np.pi), dtype=np.float64),
            'Timemin': np.array((0.0), dtype=np.float64),
            'Timemax': np.array((1.0), dtype=np.float64)
        }

        self.psipol = {
            'R': np.array((0), dtype=np.float64),
            'z': np.array((0), dtype=np.float64),
            'f': np.array((0), dtype=np.float64),
            'nPhi': np.array((1), dtype=np.int32),
            'nTime': np.array((1), dtype=np.int32),
            'Phimin': np.array((0.0), dtype=np.float64),
            'Phimax': np.array((2.0*np.pi), dtype=np.float64),
            'Timemin': np.array((0.0), dtype=np.float64),
            'Timemax': np.array((1.0), dtype=np.float64)
        }

        self.Bfield_from_shot_flag = False

    def __load_psipol_from_file(self, path: str):
        """
        Load the magnetic coordinate into the class.

        Pablo Oyola - poyola@us.es

        Note that its follow the i-HIBPsim structure:
        --> nR, nz, nPhi, nTime: int32 - Grid size in each direction.
        --> Rmin, Rmax: float64 - Minimum and maximum major radius.
        --> zmin, zmax: float64 - Minimum and maximum vertical pos.
        --> Phimin, Phimax: float64 - Min. and max. toroidal direction.
        --> Timemin, timemax> float64
        --> psipol[nR, nPhi, nz]: float64

        :param  path: Full path to the field.
        """
        entry = self.psipol
        with open(path, 'rb') as fid:
            entry['nR'] = np.fromfile(fid, 'uint32', 1)
            entry['nz'] = np.fromfile(fid, 'uint32', 1)
            entry['nPhi'] = np.fromfile(fid, 'uint32', 1)
            entry['nTime'] = np.fromfile(fid, 'uint32', 1)

            entry['Rmin'] = np.fromfile(fid, 'float64', 1)
            entry['Rmax'] = np.fromfile(fid, 'float64', 1)
            entry['zmin'] = np.fromfile(fid, 'float64', 1)
            entry['zmax'] = np.fromfile(fid, 'float64', 1)
            entry['Phimin'] = np.fromfile(fid, 'float64', 1)
            entry['Phimax'] = np.fromfile(fid, 'float64', 1)
            entry['Timemin'] = np.fromfile(fid, 'float64', 1)
            entry['Timemax'] = np.fromfile(fid, 'float64', 1)

            size2read = entry['nR'] * entry['nz'] \
                        * entry['nPhi'] * entry['nTime']

            f = np.fromfile(fid, 'float64', count=size2read[0])
            f = np.reshape(f, (entry['nR'][0], entry['nz'][0]), order='F')

        # Creating the interpolating function.
        R = np.linspace(entry['Rmin'][0], entry['Rmax'][0],
                        entry['nR'][0])
        z = np.linspace(entry['zmin'][0], entry['zmax'][0],
                        entry['nz'][0])

        self.psipol['R'] = np.array(R, dtype=np.float64)
        self.psipol['z'] = np.array(z, dtype=np.float64)
        self.psipol['Rmin'] = np.array(R.min(), dtype=np.float64)
        self.psipol['Rmax'] = np.array(R.max(), dtype=np.float64)
        self.psipol['z'] = np.array(z, dtype=np.float64)
        self.psipol['zmin'] = np.array(z.min(), dtype=np.float64)
        self.psipol['zmax'] = np.array(z.max(), dtype=np.float64)
        self.psipol['Phi'] = np.array((0, 2.0*np.pi), dtype=np.float64)
        self.psipol['Phimin'] = np.array(0.0, dtype=np.float64)
        self.psipol['Phimax'] = np.array(2.0*np.pi, dtype=np.float64)

        self.psipol['Time'] = np.array((0, 1.0), dtype=np.float64)
        self.psipol['Timemin'] = np.array(0.0, dtype=np.float64)
        self.psipol['Timemax'] = np.array(1.0, dtype=np.float64)
        self.psipol['nPhi'] = np.array([1], dtype=np.int32)
        self.psipol['nTime'] = np.array([1], dtype=np.int32)
        self.psipol['f'] = f.astype(dtype=np.float64)
        self.psipol_interp = lambda r, z, phi, time: \
            interpn((self.psipol['R'], self.psipol['z']),
                     self.psipol['f'], (r.flatten(), z.flatten()))
        self.psipol_on = True

    def readFiles(self, path: str, field_name: str):
        """
        Start the class containing the E-M fields from files.

        Pablo Oyola ft. Jose Rueda

        Note that its follow the i-HIBPsim/SINPA structure:
        --> nR, nz, nPhi, nTime: int32 - Grid size in each direction.
        --> Rmin, Rmax: float64 - Minimum and maximum major radius.
        --> zmin, zmax: float64 - Minimum and maximum vertical pos.
        --> Phimin, Phimax: float64 - Min. and max. toroidal direction.
        --> Timemin, timemax> float64
        --> Br[nR, nPhi, nz]: float64
        --> Bphi[nR, nPhi, nz]: float64
        --> Bz[nR, nPhi, nz]: float64

        :param  Bfile: Full path to the magnetic field.
        :param  Efile: Full path to the electric field file.

        :Example:
        >>> fields = fields()
        >>> fields.readFiles(Bfile='Bfield.dat', Efile='Efield.dat')
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False

        if not os.path.isfile(path):
            raise FileNotFoundError('Cannot locate the file: %s' % path)

        if field_name.lower() in ('psipol', 'rhopol', 'psi', 'rho'):
            self.__load_psipol_from_file(path)
            return

        if field_name.lower() not in ('b', 'bfield', 'e', 'efield'):
            raise ValueError('Input field name can only be B, E or rhopol')

        entry = {   'b': self.Bfield,
                    'bfield': self.Bfield,
                    'e': self.Efield,
                    'efield': self.Efield
                }.get(field_name.lower())



        with open(path, 'rb') as fid:
            entry['nR'] = np.fromfile(fid, 'uint32', 1)
            entry['nz'] = np.fromfile(fid, 'uint32', 1)
            entry['nPhi'] = np.fromfile(fid, 'uint32', 1)
            entry['nTime'] = np.fromfile(fid, 'uint32', 1)

            entry['Rmin'] = np.fromfile(fid, 'float64', 1)
            entry['Rmax'] = np.fromfile(fid, 'float64', 1)
            entry['zmin'] = np.fromfile(fid, 'float64', 1)
            entry['zmax'] = np.fromfile(fid, 'float64', 1)
            entry['Phimin'] = np.fromfile(fid, 'float64', 1)
            entry['Phimax'] = np.fromfile(fid, 'float64', 1)
            entry['Timemin'] = np.fromfile(fid, 'float64', 1)
            entry['Timemax'] = np.fromfile(fid, 'float64', 1)

            size2read = entry['nR'] * entry['nz'] \
                      * entry['nPhi'] * entry['nTime']

            fr = np.fromfile(fid, 'float64', count=size2read[0])
            fphi = np.fromfile(fid, 'float64', count=size2read[0])
            fz = np.fromfile(fid, 'float64', count=size2read[0])

        if entry['nR'] == 1:  # Just one point
            self.bdims = 0
            entry['fr'] = fr.flatten()
            entry['fz'] = fz.flatten()
            entry['ft'] = fphi.flatten()

            entry['R'] = np.linspace(entry['Rmin'][0], entry['Rmax'][0],
                                        entry['nR'][0])
            entry['z'] = np.linspace(entry['zmin'][0], entry['zmax'][0],
                                        entry['nz'][0])

            frinterp = lambda r, z, phi, time: entry['fr']
            fzinterp = lambda r, z, phi, time: entry['fz']
            fphiinterp = lambda r, z, phi, time: entry['ft']

        elif entry['nPhi'] == 1 and entry['nTime'] == 1:
            self.bdims = 2  # Static 2D fields
            entry['fr'] = fr.reshape((entry['nR'][0],
                                        entry['nz'][0]),
                                            order='F')
            entry['fz'] = fz.reshape((entry['nR'][0],
                                            entry['nz'][0]),
                                            order='F')
            entry['ft'] = fphi.reshape((entry['nR'][0],
                                                entry['nz'][0]),
                                                order='F')

            entry['R'] = np.linspace(entry['Rmin'][0],
                                            entry['Rmax'][0],
                                            entry['nR'][0])
            entry['z'] = np.linspace(entry['zmin'][0],
                                            entry['zmax'][0],
                                            entry['nz'][0])

            frinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['z']),
                        entry['fr'], (r.flatten(), z.flatten()))
            fzinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['z']),
                        entry['fz'], (r.flatten(), z.flatten()))
            fphiinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['z']),
                        entry['ft'], (r.flatten(), z.flatten()))

        elif entry['nTime'] == 1:
            self.bdims = 3  # Static 3D fields
            entry['fr'] = fr.reshape((entry['nR'][0],
                                        entry['nPhi'][0],
                                        entry['nz'][0]),
                                        order='F')
            entry['fz'] = fz.reshape((entry['nR'][0],
                                        entry['nPhi'][0],
                                        entry['nz'][0]),
                                        order='F')
            entry['ft'] = fphi.reshape((entry['nR'][0],
                                        entry['nPhi'][0],
                                        entry['nz'][0]),
                                        order='F')

            entry['R'] = np.linspace(entry['Rmin'][0],
                                        entry['Rmax'][0],
                                        entry['nR'][0])
            entry['z'] = np.linspace(entry['zmin'][0],
                                        entry['zmax'][0],
                                        entry['nz'][0])
            entry['Phi'] = np.linspace(entry['Phimin'][0],
                                        entry['Phimax'][0],
                                        entry['nPhi'][0])

            frinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['Phi'],
                            entry['z']), entry['fr'],
                        (r.flatten(), phi.flatten(), z.flatten()))
            fzinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['Phi'],
                            entry['z']), entry['fz'],
                        (r.flatten(), phi.flatten(), z.flatten()))
            fphiinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['Phi'],
                            entry['z']), entry['ft'],
                        (r.flatten(), phi.flatten(), z.flatten()))

        else:
            self.bdims = 4  # Full 4D field
            entry['fr'] = fr.reshape((entry['nR'][0],
                                            entry['nPhi'][0],
                                            entry['nz'][0],
                                            entry['nTime'][0]),
                                            order='F')
            entry['fz'] = fz.reshape((entry['nR'][0],
                                            entry['nPhi'][0],
                                            entry['nz'][0],
                                            entry['nTime'][0]),
                                            order='F')
            entry['ft'] = fphi.reshape((entry['nR'][0],
                                                entry['nPhi'][0],
                                                entry['nz'][0],
                                                entry['nTime'][0]),
                                                order='F')

            entry['R'] = np.linspace(entry['Rmin'][0],
                                            entry['Rmax'][0],
                                            entry['nR'][0])
            entry['z'] = np.linspace(entry['zmin'][0],
                                            entry['zmax'][0],
                                            entry['nz'][0])
            entry['Phi'] = np.linspace(entry['Phimin'][0],
                                                entry['Phimax'][0],
                                                entry['nPhi'][0])
            entry['time'] = \
                np.linspace(entry['Timemin'][0],
                            entry['Timemax'][0],
                            entry['nTime'][0])

            frinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['Phi'],
                            entry['z'], entry['time']),
                        entry['fr'],
                        (r.flatten(), phi.flatten(),
                            z.flatten(), time.flatten()))
            fzinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['Phi'],
                            entry['z'], entry['time']),
                        entry['fz'],
                        (r.flatten(), phi.flatten(),
                            z.flatten(), time.flatten()))
            fphiinterp = lambda r, z, phi, time: \
                interpn((entry['R'], entry['Phi'],
                            entry['z'], entry['time']),
                        entry['ft'],
                        (r.flatten(), phi.flatten(),
                            z.flatten(), time.flatten()))

        # Associating the data with the interpolators.
        if field_name.lower() in ('b', 'bfield'):
            self.Brinterp = frinterp
            self.Bzinterp = fzinterp
            self.Bphiinterp = fphiinterp
        else:
            self.Erinterp = frinterp
            self.Ezinterp = fzinterp
            self.Ephiinterp = fphiinterp

    def readBfromDB(self, shotnumber: int = 34570, time: float = 2.5,
                    exp: str = 'AUGD', diag: str = 'EQI',
                    edition: int = 0,
                    Rmin: float = 1.03, Rmax: float = 2.65,
                    zmin: float = -1.224, zmax: float = 1.05,
                    nR: int = 128, nz: int = 256,
                    readPsi: bool = True):
        """
        Read field from machine database.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        ft.
        Jose Rueda: jrrueda@us.es

        :param  shotnumber: Shot from which to extract the magnetic equilibrium.
        :param  time: Time point to fetch the equilibrium.
        :param  exp: Experiment where the equilibria is stored.
        :param  diag: Diagnostic from which extracting the equilibrium.
        :param  edition: Edition of the equilibrium to retrieve. Set to 0 by
        default, which will take from the AUG DB the latest version.
        :param  Rmin: Minimum radius to get the magnetic equilibrium.
        :param  Rmax: Maximum radius to get the magnetic equilibrium.
        :param  zmin: Minimum Z to get the magnetic equilibrium.
        :param  zmax: Maximum Z to get the magnetic equilibrium.
        :param  nR: Number of points to define the B field grid in R direction.
        :param  nz: Number of points to define the B field grid in Z direction.
        :param readPsi: if true, read also psi pol from the database
        
        :Example:
        >>> import ScintSuite.SimulationCodes.Common.fields as fields
        >>> fields = fields()
        >>> fields.readBfromDB(shotnumber=34570, time=2.5, exp='AUGD',
        >>>                    diag='EQI', edition=0, Rmin=1.03, Rmax=2.65,
        >>>                    zmin=-1.224, zmax=1.05, nR=128, nz=256)

        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False

        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        br, bz, bt, bp = ssdat.get_mag_field(shotnumber, RR.flatten(),
                                        zz.flatten(),
                                        exp=exp,
                                        ed=edition,
                                        diag=diag,
                                        time=time)

        del RR
        del zz
        del bp
        Br = np.asfortranarray(np.reshape(br, grid_shape).T)
        Bz = np.asfortranarray(np.reshape(bz, grid_shape).T)
        Bt = np.asfortranarray(np.reshape(bt, grid_shape).T)
        del br
        del bt
        del bz

        # Storing the data in the class.
        self.bdims = 2
        self.Bfield['R'] = np.array(R, dtype=np.float64)
        self.Bfield['z'] = np.array(z, dtype=np.float64)
        self.Bfield['Rmin'] = np.array((Rmin), dtype=np.float64)
        self.Bfield['Rmax'] = np.array((Rmax), dtype=np.float64)
        self.Bfield['zmin'] = np.array((zmin), dtype=np.float64)
        self.Bfield['zmax'] = np.array((zmax), dtype=np.float64)
        self.Bfield['nR'] = np.array([nR], dtype=np.int32)
        self.Bfield['nz'] = np.array([nz], dtype=np.int32)
        self.Bfield['fr'] = Br.astype(dtype=np.float64)
        self.Bfield['fz'] = Bz.astype(dtype=np.float64)
        self.Bfield['ft'] = Bt.astype(dtype=np.float64)

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (np.atleast_1d(r).flatten(), np.atleast_1d(z).flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (np.atleast_1d(r).flatten(), np.atleast_1d(z).flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (np.atleast_1d(r).flatten(), np.atleast_1d(z).flatten()))

        # Retrieving as well the poloidal magnetic flux.
        if readPsi:
            self.readPsiPolfromDB(time=time, shotnumber=shotnumber,
                                exp=exp, diag=diag, edition=edition,
                                Rmin=Rmin, Rmax=Rmax,
                                zmin=zmin, zmax=zmax,
                                nR=nR, nz=nz)

        # Saving the input data to the class.
        self.Bfield_from_shot_flag = True
        self.shotnumber = shotnumber
        self.timepoint = time
        if machine != 'MU':
            self.diag = diag
            self.exp = exp
            self.edition = edition

    def readBfromDBSinglePoint(self, shotnumber: int = 39612,
                               time: float = 2.5, R0: float = 1.90,
                               z0: float = 0.9,
                               exp: str = 'AUGD', diag: str = 'EQI',
                               edition: int = 0,
                               Rmin: float = 1.60, Rmax: float = 2.20,
                               zmin: float = 0.8, zmax: float = 1.200,
                               nR: int = 40, nz: int = 80):
        """
        Read field from machine database, single point

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        ft.
        Jose Rueda: jrrueda@us.es

        The idea is that it will take the magnetic field in a single point and
        then use this value in all points in the grid. Notice that we will not
        create an uniform grid in this way because we will be putting the same
        2D grid, so we will be creating a 'toroidally homogeneous field' (if
        this term exist).

        :param  shotnumber: Shot from which to extract the magnetic equilibrium.
        :param  time: Time point to fetch the equilibrium.
        :param  R0: R where to calculate the magnetic field [m]
        :param  Z0: Z where to calculate the
        :param  exp: Experiment where the equilibria is stored.
        :param  diag: Diagnostic from which extracting the equilibrium.
        :param  edition: Edition of the equilibrium to retrieve. Set to 0 by
        default, which will take from the AUG DB the latest version.
        :param  Rmin: Minimum radius to get the magnetic equilibrium.
        :param  Rmax: Maximum radius to get the magnetic equilibrium.
        :param  zmin: Minimum Z to get the magnetic equilibrium.
        :param  zmax: Maximum Z to get the magnetic equilibrium.
        :param  nR: Number of points to define the B field grid in R direction.
        :param  nz: Number of points to define the B field grid in Z direction.
        
        :Example:
        >>> import ScintSuite.SimulationCodes.Common.fields as fields
        >>> fields = fields()
        >>> fields.readBfromDBSinglePoint(shotnumber=34570, time=2.5,
        >>>                               R0=1.9, z0=0.92, exp='AUGD',
        >>>                               diag='EQI', edition=0, Rmin=1.6,
        >>>                               Rmax=2.0, zmin=0.8, zmax=1.2,
        >>>                               nR=40, nz=80)

        """
        self.bdims = 0

        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        br, bz, bt, bp = ssdat.get_mag_field(shotnumber, RR.flatten(),
                                             zz.flatten(),
                                             exp=exp,
                                             ed=edition,
                                             diag=diag,
                                             time=time)
        del RR
        del zz
        del bp
        Br = np.asfortranarray(np.reshape(br, grid_shape).T)
        Bz = np.asfortranarray(np.reshape(bz, grid_shape).T)
        Bt = np.asfortranarray(np.reshape(bt, grid_shape).T)
        del br
        del bt
        del bz

        # Storing the data in the class.
        self.bdims = 2
        self.Bfield['R'] = np.array(R * 100.0, dtype=np.float64)
        self.Bfield['z'] = np.array(z * 100.0, dtype=np.float64)
        self.Bfield['Rmin'] = np.array((Rmin), dtype=np.float64)
        self.Bfield['Rmax'] = np.array((Rmax), dtype=np.float64)
        self.Bfield['zmin'] = np.array((zmin), dtype=np.float64)
        self.Bfield['zmax'] = np.array((zmax), dtype=np.float64)
        self.Bfield['nR'] = np.array([nR], dtype=np.int32)
        self.Bfield['nz'] = np.array([nz], dtype=np.int32)
        self.Bfield['fr'] = Br.astype(dtype=np.float64)
        self.Bfield['fz'] = Bz.astype(dtype=np.float64)
        self.Bfield['ft'] = Bt.astype(dtype=np.float64)

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        # Calculate the value of the field in the desired point
        rr0 = np.array(R0)
        zz0 = np.array(z0)
        bbr = self.Brinterp(rr0, zz0, 0.0, 0.0)
        bbz = self.Bzinterp(rr0, zz0, 0.0, 0.0)
        bbphi = self.Bphiinterp(rr0, zz0, 0.0, 0.0)

        # Impose this value in all the grid points:
        self.Bfield['fr'][:] = bbr
        self.Bfield['fz'][:] = bbz
        self.Bfield['ft'][:] = bbphi

        # Re-create the interpolators
        self.Brinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        # Saving the input data to the class.
        self.Bfield_from_shot_flag = True
        self.shotnumber = shotnumber
        self.edition = edition
        self.timepoint = time
        self.diag = diag
        self.exp = exp

    def createFromSingleB(self, B: np.ndarray, Rmin: float = 1.6,
                          Rmax: float = 2.2,
                          zmin: float = 0.8, zmax: float = 1.2,
                          nR: int = 40, nz: int = 80):
        """
        Create a field for SINPA from a given B vector

        Jose Rueda: jrrueda@us.es

        The idea is that it will take the given magnetic field and
        then use this value in all points in the grid. Notice that we will not
        create an uniform grid in this way because we will be putting the same
        2D grid, so we will be creating a 'toroidally homogeneous field' (if
        this term exist).

        :param  B: Magnetic field to be used [Br,Bz,Bphi] [T]
        :param  Rmin: Minimum radius to get the magnetic equilibrium.
        :param  Rmax: Maximum radius to get the magnetic equilibrium.
        :param  zmin: Minimum Z to get the magnetic equilibrium.
        :param  zmax: Maximum Z to get the magnetic equilibrium.
        :param  nR: Number of points to define the B field grid in R direction.
        :param  nz: Number of points to define the B field grid in Z direction.

        :Example:
        >>> import ScintSuite.SimulationCodes.Common.fields as fields
        >>> fields = fields()
        >>> fields.createFromSingleB(B=[0.0, 0.0, 1.0], Rmin=1.6, Rmax=2.0,
        >>>                          zmin=0.8, zmax=1.2, nR=40, nz=80)

        """
        self.bdims = 0

        # Prepare the grid and allocate the field
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        br = np.zeros(grid_shape)
        bz = np.zeros(grid_shape)
        bt = np.zeros(grid_shape)

        del RR
        del zz
        Br = np.asfortranarray(br.T)
        Bz = np.asfortranarray(bz.T)
        Bt = np.asfortranarray(bt.T)
        del br
        del bt
        del bz
        # Save the values
        Br[:] = B[0]
        Bz[:] = B[1]
        Bt[:] = B[2]

        # Storing the data in the class.
        self.bdims = 2
        self.Bfield['R'] = np.array(R, dtype=np.float64)
        self.Bfield['z'] = np.array(z, dtype=np.float64)
        self.Bfield['Rmin'] = np.array((Rmin), dtype=np.float64)
        self.Bfield['Rmax'] = np.array((Rmax), dtype=np.float64)
        self.Bfield['zmin'] = np.array((zmin), dtype=np.float64)
        self.Bfield['zmax'] = np.array((zmax), dtype=np.float64)
        self.Bfield['nR'] = np.array([nR], dtype=np.int32)
        self.Bfield['nz'] = np.array([nz], dtype=np.int32)
        self.Bfield['fr'] = Br.astype(dtype=np.float64)
        self.Bfield['fz'] = Bz.astype(dtype=np.float64)
        self.Bfield['ft'] = Bt.astype(dtype=np.float64)

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        # Saving the input data to the class.
        self.Bfield_from_shot_flag = False

    def createHomogeneousField(self, F: np.ndarray, field: str = 'B'):
        """
        Create a field for SINPA from a given B vector

        Jose Rueda: jrrueda@us.es

        Create an homogenous field for SINPA

        @todo: include interpolators, maybe translation to polar??

        :param  F: array with field, [fx, fy, fz]
        :param  field: 'B' or 'E', the field you want to generate

        :Example:
        >>> import ScintSuite.SimulationCodes.Common.fields as fields
        >>> fields = fields()
        >>> fields.createHomogeneousField(F=[0.0, 0.0, 1.0], field='B')

        """
        if field.lower() == 'b':
            self.bdims = 0
            key = 'Bfield'
            self.Bfield_from_shot_flag = False

        elif field.lower() == 'e':
            self.edims = 0
            key = 'Efield'
        else:
            raise Exception('Field not understood!')
        # Store the field
        self.__dict__[key]['R'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['z'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['Rmin'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['Rmax'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['zmin'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['zmax'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['nR'] = np.array([1], dtype=np.int32)
        self.__dict__[key]['nz'] = np.array([1], dtype=np.int32)
        self.__dict__[key]['fx'] = np.array(F[0]).astype(dtype=np.float64)
        self.__dict__[key]['fy'] = np.array(F[1]).astype(dtype=np.float64)
        self.__dict__[key]['fz'] = np.array(F[2]).astype(dtype=np.float64)

    def createHomogeneousFieldThetaPhi(self, theta: float, phi: float,
                                       field_mod: float = 1.0,
                                       field: str = 'B',
                                       u1=np.array((1.0, 0.0, 0.0)),
                                       u2=np.array((0.0, 1.0, 0.0)),
                                       u3=np.array((0.0, 0.0, 1.0)),
                                       IpBt_sign: float = -1.0,
                                       verbose: bool = True,
                                       diagnostic: str = 'FILD'):
        """
        Create a field for SINPA from a given B vector

        Jose Rueda: jrrueda@us.es

        Create an homogenous field for SINPA, defined by the theta and phi
        direction angles

        @todo: include interpolators, maybe translation to polar??

        :param  theta: Theta angle, in degrees, as defined in FILDSIM. See
            extended documentation
        :param  phi: Phi angle, in degrees, as defined in FILDSIM. See
            extended documentation
        :param  field_mod: modulus of the field to generate
        :param  field: 'B' or 'E', the field you want to generate
        :param  u1: u1 vector of the reference system
        :param  u2: u2 vector for the referece system
        :param  u3: u3 vector for the reference system
        :param  diagnostic: string identifying the diagnostic (to determine the
            meaining of the angles)

        Note: Please see SINPA documentation for a nice drawing of the
        different angles

        :Example:
        >>> import ScintSuite.SimulationCodes.Common.fields as fields
        >>> fields = fields()
        >>> fields.createHomogeneousFieldThetaPhi(theta=0.0, phi=0.0,
        >>>                                      field_mod=1.0, field='B',
        >>>                                      u1=np.array((1.0, 0.0, 0.0)),
        >>>                                      u2=np.array((0.0, 1.0, 0.0)),
        >>>                                      u3=np.array((0.0, 0.0, 1.0)),
        >>>                                      IpBt_sign=-1.0,
        >>>                                      verbose=True,
        >>>                                      diagnostic='FILD')

        """
        # --- Set field flags:
        if field.lower() == 'b':
            self.bdims = 0
            key = 'Bfield'
            self.Bfield_from_shot_flag = False

        elif field.lower() == 'e':
            self.edims = 0
            key = 'Efield'
        else:
            raise Exception('Field not understood!')
        # --- Generate the direction
        t = theta * np.pi / 180.0  # pass to radians
        p = phi * np.pi / 180.0
        if diagnostic.lower() == 'fild':
            dir = math.sin(p) * u1 + math.cos(p)\
                * (math.cos(t) * u2 - math.sin(t) * u3)
            F = IpBt_sign * field_mod * dir
        elif diagnostic.lower() == 'inpa':
            dir = math.sin(t) * (math.cos(p) * u1 + math.sin(p) * u2)\
                + math.cos(t) * u3
            F = field_mod * dir
        else:
            raise errors.NotValidInput('Not understood diagnostic')

        if verbose:
            print('Value of the field is: ', F)
        # --- Store the field
        self.__dict__[key]['R'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['z'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['Rmin'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['Rmax'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['zmin'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['zmax'] = np.array((0.0), dtype=np.float64)
        self.__dict__[key]['nR'] = np.array([1], dtype=np.int32)
        self.__dict__[key]['nz'] = np.array([1], dtype=np.int32)
        self.__dict__[key]['fx'] = np.array(F[0]).astype(dtype=np.float64)
        self.__dict__[key]['fy'] = np.array(F[1]).astype(dtype=np.float64)
        self.__dict__[key]['fz'] = np.array(F[2]).astype(dtype=np.float64)

    def readPsiPolfromDB(self, shotnumber: int = 34570, time: float = 2.5,
                         exp: str = 'AUGD', diag: str = 'EQI',
                         edition: int = 0,
                         Rmin: float = 1.03, Rmax: float = 2.65,
                         zmin: float = -1.224, zmax: float = 1.05,
                         nR: int = 128, nz: int = 256,
                         coord_type: str='psipol'):
        """
        Fetchs the psi_pol = rho_pol(R, z) map from the AUG database using
        input grid.

        Jose Rueda: jrrueda@us.es
        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  time: Time point from which magnetic equilibrium will be read.
        :param  shotnumber: Shot from which to extract the magnetic equilibria
        :param  exp: Experiment where the equilibria is stored.
        :param  diag: Diagnostic from which extracting the equilibrium.
        :param  edition: Edition of the equilibrium to retrieve. Set to 0 by
        default, which will take from the AUG DB the latest version.
        :param  Rmin: Minimum radius to get the magnetic equilibrium.
        :param  Rmax: Maximum radius to get the magnetic equilibrium.
        :param  zmin: Minimum Z to get the magnetic equilibrium.
        :param  zmax: Maximum Z to get the magnetic equilibrium.
        :param  nR: Number of points to define the B field grid in R direction.
        :param  nz: Number of points to define the B field grid in Z direction.
        
        :Example:
        >>> import ScintSuite.SimulationCodes.Common.fields as fields
        >>> fields = fields()
        >>> fields.readPsiPolfromDB(shotnumber=34570, time=2.5, exp='AUGD',
        >>>                          diag='EQI', edition=0, Rmin=1.03,
        >>>                          Rmax=2.65, zmin=-1.224, zmax=1.05,
        >>>                          nR=128, nz=256)

        """
        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        if coord_type == 'psipol':
            psipol = ssdat.get_psipol(shotnumber, RR.flatten(), zz.flatten(),
                                      diag=diag, time=time, exp=exp)
    
            # Reshaping into the original shape.
            psipol = np.reshape(psipol, grid_shape).T
            self.psipol_type = 'psipol'
        elif coord_type == 'rhopol':
            print('getting rhopol')
            psipol = ssdat.get_rho(shotnumber, RR.flatten(), zz.flatten(),
                                   diag=diag, time=time, exp=exp,
                                   coord_out='rho_pol')
            psipol = np.reshape(psipol, grid_shape).T
            self.psipol_type = 'rhopol'
        elif coord_type == 'rhotor':
            psipol = ssdat.get_rho(shotnumber, RR.flatten(), zz.flatten(),
                                   diag=diag, time=time, exp=exp,
                                   coord_out='rho_tor')
            psipol = np.reshape(psipol, grid_shape).T
            self.psipol_type = 'rhotor'
        else:
            raise ValueError(f'Magnetic coordinate {coord_type} not recognized')

        # Creating the interpolating function.
        self.psipol['R'] = np.array(R, dtype=np.float64)
        self.psipol['z'] = np.array(z, dtype=np.float64)
        self.psipol['Rmin'] = np.array(R.min(), dtype=np.float64)
        self.psipol['Rmax'] = np.array(R.max(), dtype=np.float64)
        self.psipol['z'] = np.array(z, dtype=np.float64)
        self.psipol['zmin'] = np.array(z.min(), dtype=np.float64)
        self.psipol['zmax'] = np.array(z.max(), dtype=np.float64)
        self.psipol['Phi'] = np.array((0, 2.0*np.pi), dtype=np.float64)
        self.psipol['Phimin'] = np.array(0.0, dtype=np.float64)
        self.psipol['Phimax'] = np.array(2.0*np.pi, dtype=np.float64)

        self.psipol['Time'] = np.array((0, 1.0), dtype=np.float64)
        self.psipol['Timemin'] = np.array(0.0, dtype=np.float64)
        self.psipol['Timemax'] = np.array(1.0, dtype=np.float64)
        self.psipol['nR'] = np.array([nR], dtype=np.int32)
        self.psipol['nz'] = np.array([nz], dtype=np.int32)
        self.psipol['nPhi'] = np.array([1], dtype=np.int32)
        self.psipol['nTime'] = np.array([1], dtype=np.int32)
        self.psipol['f'] = psipol.astype(dtype=np.float64)
        self.psipol['Rmin'] = np.array((Rmin), dtype=np.float64)
        self.psipol['Rmax'] = np.array((Rmax), dtype=np.float64)
        self.psipol['zmin'] = np.array((zmin), dtype=np.float64)
        self.psipol['zmax'] = np.array((zmax), dtype=np.float64)
        self.psipol_interp = lambda r, z, phi, time: \
            interpn((self.psipol['R'], self.psipol['z']),
                    self.psipol['f'], (r.flatten(), z.flatten()))
        self.psipol_on = True
        return

    def getBfield(self, R: float, z: float,
                  phi: float = None, t: float = None):
        """
        Get the magnetic field components at the given points.

        Note, it extract the field using the interpolators

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  R: Major radius to evaluate the magnetic field.
        :param  z: Vertical position to evaluate the magnetic field.
        :param  phi: Toroidal location to evaluate the magnetic field.
        If the system is in 2D, it will be ignored.
        :param  t: Time to evaluate the magnetic field. If the magnetic
        field is only stored for a single time, it will be ignored.

        :return Br: Radial component of the magnetic field.
        :return Bz: Vertical component of the magnetic field.
        :return Bphi: Toroidal component of the magnetic field.

        :Example:
        >>> # Assuming that the field has been loaded in the class
        >>> field.getBfield(R=1.9, z=0.92, phi=0.0, t=2.5)
        """
        if self.bdims != 0:
            Br = self.Brinterp(R, z, phi, t)
            Bz = self.Bzinterp(R, z, phi, t)
            Bphi = self.Bphiinterp(R, z, phi, t)
        else:
            raise Exception('The magnetic field has not been loaded!')

        return Br, Bz, Bphi

    def getEfield(self, R: float, z: float,
                  phi: float = None, t: float = None):
        """
        Get the electric field components at the given points.

        Note, it extract the field using the interpolators

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  R: Major radius to evaluate the electric field.
        :param  z: Vertical position to evaluate the electric field.
        :param  phi: Toroidal location to evaluate the electric field.
        If the system is in 2D, it will be ignored.
        :param  t: Time to evaluate the electric field. If the electric
        field is only stored for a single time, it will be ignored.

        :return Er: Radial component of the magelectricnetic field.
        :return Ez: Vertical component of the electric field.
        :return Ephi: Toroidal component of the electric field.

        :Example:
        >>> # Assuming that the field has been loaded in the class
        >>> field.getEfield(R=1.9, z=0.92, phi=0.0, t=2.5)
        """
        if self.edims != 0:
            Er = self.Erinterp(R, z, phi, t)
            Ez = self.Ezinterp(R, z, phi, t)
            Ephi = self.Ephiinterp(R, z, phi, t)
        else:
            raise Exception('The electric field has not been loaded!')

        return Er, Ez, Ephi

    def getPsipol(self, R: float, z: float,
                  phi: float = None, t: float = None):
        """
        Get the poloidal magnetic flux at the position.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  R: Major radius to evaluate the poloidal flux.
        :param  z: Vertical position to evaluate the poloidal flux.
        :param  phi: Toroidal location to evaluate the poloidal flux.
        If the system is in 2D, it will be ignored.
        :param  t: Time to evaluate the poloidal flux. If the poloidal
        flux is only stored for a single time, it will be ignored.

        :return psipol: Poloidal flux at the input points.

        :Example:
        >>> # Assuming that the field has been loaded in the class
        >>> field.getPsipol(R=1.9, z=0.92, phi=0.0, t=2.5)

        """
        if self.psipol_on:
            psipol = self.psipol_interp(R, z, phi, t)
        else:
            raise Exception('The poloidal flux is not loaded!')

        return psipol

    def tofile(self, fid, what: str='Bfield'):
        """
        Write the field to files following the i-HIBPsims scheme.

        Pablo Oyola - pablo.oyola@ipp.mpg.de ft. jrrueda@us.es

        :param  fid: file identifier where the files will be written. if this is
            a string, the file will be created.
        :param  bflag: states if the magnetic field has to be written.
        Default is True, so magnetic field will be written.
        :param  eflag: states if the electric field has to be written.
        Default to False. If this is set to True, the magnetic field
        will not be written.

        :Example:
        >>> # Assuming that the field has been loaded in the class
        >>> field.tofile(fid='field.bin', what='Bfield')
        
        """
        if isinstance(fid, str):
            fid = open(fid, 'wb')
            opened = True
        else:
            opened = False

        if what.lower() == 'bfield':
            # Write header with grid size information:
            np.array(self.Bfield['nR'], dtype='int32').tofile(fid)
            np.array(self.Bfield['nz'], dtype='int32').tofile(fid)
            np.array(self.Bfield['nPhi'], dtype='int32').tofile(fid)
            np.array(self.Bfield['nTime'], dtype='int32').tofile(fid)

            # Write grid ends:
            np.array(self.Bfield['Rmin'], dtype='float64').tofile(fid)
            np.array(self.Bfield['Rmax'], dtype='float64').tofile(fid)
            np.array( self.Bfield['zmin'], dtype='float64').tofile(fid)
            np.array(self.Bfield['zmax'], dtype='float64').tofile(fid)
            np.array(self.Bfield['Phimin'], dtype='float64').tofile(fid)
            np.array(self.Bfield['Phimax'], dtype='float64').tofile(fid)
            np.array(self.Bfield['Timemin'], dtype='float64').tofile(fid)
            np.array(self.Bfield['Timemax'], dtype='float64').tofile(fid)

            # Write fields
            if self.bdims > 0:
                self.Bfield['fr'].ravel(order='F').tofile(fid)
                self.Bfield['ft'].ravel(order='F').tofile(fid)
                self.Bfield['fz'].ravel(order='F').tofile(fid)
            else:
                self.Bfield['fx'].ravel(order='F').tofile(fid)
                self.Bfield['fy'].ravel(order='F').tofile(fid)
                self.Bfield['fz'].ravel(order='F').tofile(fid)
        elif what.lower() == 'efield':
            # Write header with grid size information:
            self.Efield['nR'].tofile(fid)
            self.Efield['nz'].tofile(fid)
            self.Efield['nPhi'].tofile(fid)
            self.Efield['nTime'].tofile(fid)

            # Write grid ends:
            self.Efield['Rmin'].tofile(fid)
            self.Efield['Rmax'].tofile(fid)
            self.Efield['zmin'].tofile(fid)
            self.Efield['zmax'].tofile(fid)
            self.Efield['Phimin'].tofile(fid)
            self.Efield['Phimax'].tofile(fid)
            self.Efield['Timemin'].tofile(fid)
            self.Efield['Timemin'].tofile(fid)

            # Write fields
            if self.edims > 0:
                self.Efield['fr'].ravel(order='F').tofile(fid)
                self.Efield['ft'].ravel(order='F').tofile(fid)
                self.Efield['fz'].ravel(order='F').tofile(fid)
            else:
                self.Efield['fx'].ravel(order='F').tofile(fid)
                self.Efield['fy'].ravel(order='F').tofile(fid)
                self.Efield['fz'].ravel(order='F').tofile(fid)
        elif what.lower() == 'psipol':
            np.array(self.psipol['nR'], dtype='int32').tofile(fid)
            np.array(self.psipol['nz'], dtype='int32').tofile(fid)
            np.array(self.psipol['nPhi'], dtype='int32').tofile(fid)
            np.array(self.psipol['nTime'], dtype='int32').tofile(fid)

            # Write grid ends:
            np.array(self.psipol['Rmin'], dtype='float64').tofile(fid)
            np.array(self.psipol['Rmax'], dtype='float64').tofile(fid)
            np.array(self.psipol['zmin'], dtype='float64').tofile(fid)
            np.array(self.psipol['zmax'], dtype='float64').tofile(fid)
            np.array(self.psipol['Phimin'], dtype='float64').tofile(fid)
            np.array(self.psipol['Phimax'], dtype='float64').tofile(fid)
            np.array(self.psipol['Timemin'], dtype='float64').tofile(fid)
            np.array(self.psipol['Timemin'], dtype='float64').tofile(fid)

            # Write fields
            self.psipol['f'].astype(dtype='float64').ravel(order='F').tofile(fid)
        else:
            raise ValueError('Not a valid field to be written.')
        if opened:
            fid.close()

    def from_netcdf(self, fn: str, rmin: float=None, rmax: float=None,
                    zmin: float=None, zmax: float=None):
        """
        Import the magnetic field from a netCDF4 file.

        Pablo Oyola - poyola@us.es
        """
        if not os.path.isfile(fn):
            raise FileNotFoundError('Cannot find %s'%fn)

        with nc.Dataset(fn, 'r') as root:
            R = np.array(root.variables['R'][:])
            z = np.array(root.variables['z'][:])
            Br = np.array(root.variables['Br'][:])
            Bz = np.array(root.variables['Bz'][:])
            Bphi = np.array(root.variables['Bphi'][:])
            rhop = np.array(root.variables['rhop'][:])

        # We check now whether we have to cut the domain limits.
        r_flags = np.ones_like(R, dtype=bool)
        z_flags = np.ones_like(z, dtype=bool)

        if rmin is not None:
            r_flags = r_flags & (R > rmin)
        if rmax is not None:
            r_flags = r_flags & (R < rmax)

        if zmin is not None:
            z_flags = z_flags & (z > zmin)
        if zmax is not None:
            z_flags = z_flags & (z < zmax)

        # Cutting the data.
        R = R[r_flags]
        z = z[z_flags]

        Br = Br[r_flags, :]
        Br = Br[:, z_flags]
        Bz = Bz[r_flags, :]
        Bz = Bz[:, z_flags]
        Bphi = Bphi[r_flags, :]
        Bphi = Bphi[:, z_flags]
        rhop = rhop[r_flags, :]
        rhop = rhop[:, z_flags]



        # With the data read from the netCDF4 file, we allocate the internal
        # data.
        self.Bfield['R'] = R.astype(dtype=np.float64)
        self.Bfield['z'] = z.astype(dtype=np.float64)
        self.Bfield['nR'] = np.array((len(R)), dtype=np.int32)
        self.Bfield['nz'] = np.array((len(z)), dtype=np.int32)
        self.Bfield['nPhi'] = np.array((1), dtype=np.int32)
        self.Bfield['nTime'] = np.array((1), dtype=np.int32)

        self.Bfield['Rmin'] = np.array((R.min()), dtype=np.float64)
        self.Bfield['Rmax'] = np.array((R.max()), dtype=np.float64)
        self.Bfield['zmin'] = np.array((z.min()), dtype=np.float64)
        self.Bfield['zmax'] = np.array((z.max()), dtype=np.float64)
        self.Bfield['Phimin'] = np.array((0.0), dtype=np.float64)
        self.Bfield['Phimax'] = np.array((2.0*np.pi), dtype=np.float64)
        self.Bfield['Timemin'] = np.array((0.0), dtype=np.float64)
        self.Bfield['Timemax'] = np.array((1.0), dtype=np.float64)

        self.Bfield['fr'] = Br.astype(dtype=np.float64)
        self.Bfield['fz'] = Bz.astype(dtype=np.float64)
        self.Bfield['ft'] = Bphi.astype(dtype=np.float64)

        self.bdims = 2

    def plot(self, fieldName: str, phiSlice: int = None, timeSlice: int = None,
             ax_options: dict = {}, ax=None, cmap=None, nLevels: int = 50,
             cbar_tick_format: str = '%.2e', plot_vessel: bool = True):
        """
        Plot the input field.

        Plots the input field ('Br', 'Bz', 'Bphi', 'Er', 'Ez', 'Ephi',
        'B', 'E' or ''Psipol') into some axis, ax, or the routine creates one
        for the plotting.

        :param  fieldName: Name of the field to be plotted.
        :param  ax_options: options for the function axis_beauty.
        :param  ax: Axis to plot the data.
        :param  cmap: Colormap to use. Gamma_II is set by default.
        :param  ax: Return the axis where the plot has been done.
        :param  plot_vessel: Flag to plot the vessel
        """
        # --- Initialise the plotting parameters
        ax_params = {
            'grid': 'both',
            'ratio': 'equal'
        }
        ax_options.update(ax_params)
        if cmap is None:
            ccmap = ssplt.Gamma_II()
        else:
            ccmap = cmap

        # If there is not any axis provided, create one.
        if ax is None:
            fig, ax = plt.subplots()

        # Change to lower case in order to make the choice more flexible.
        fieldName = fieldName.lower()

        # Raise an exception if the corresponding field is not loaded.
        dims = 2
        if fieldName.startswith('e'):
            if self.edims == 0:
                raise Exception('Plot not working of 0D fields')
            dims = self.edims
            if dims >= 3:
                phiMaxIdx = self.Efield['nPhi']
                if dims >= 4:
                    timeMaxIdx = self.Efield['nTime']

        if fieldName.startswith('b'):
            if self.bdims == 0:
                raise Exception('Plot not working of 0D fields')
            dims = self.bdims
            if dims >= 3:
                phiMaxIdx = self.Bfield['nPhi']
                if dims >= 4:
                    timeMaxIdx = self.Bfield['nTime']

        if fieldName == 'psipol' and not self.psipol_on:
            raise Exception('Magnetic flux is not loaded!')

        # Get the appropriate field
        field = {
            'br': self.Bfield['fr'],
            'bz': self.Bfield['fz'],
            'bphi': self.Bfield['ft'],
            'er': self.Efield['fr'],
            'ez': self.Efield['fz'],
            'ephi': self.Efield['ft'],
            'b': np.sqrt(self.Bfield['fr']**2
                         + self.Bfield['fz']**2
                         + self.Bfield['ft']**2),
            'e': np.sqrt(self.Efield['fr']**2
                         + self.Efield['fz']**2
                         + self.Efield['ft']**2),
            'psipol': self.psipol['f']
        }.get(fieldName.lower())

        Rplot = {
            'br': self.Bfield['R'],
            'bz': self.Bfield['R'],
            'bphi': self.Bfield['R'],
            'er': self.Efield['R'],
            'ez': self.Efield['R'],
            'ephi': self.Efield['R'],
            'b': self.Bfield['R'],
            'e': self.Efield['R'],
            'psipol': self.psipol['R']
        }.get(fieldName.lower())

        Zplot = {
            'br': self.Bfield['z'],
            'bz': self.Bfield['z'],
            'bphi': self.Bfield['z'],
            'er': self.Efield['z'],
            'ez': self.Efield['z'],
            'ephi': self.Efield['z'],
            'b': self.Bfield['z'],
            'e': self.Efield['z'],
            'psipol': self.psipol['z']
        }.get(fieldName.lower())

        # For the 3D case, only a 2D projection can be plotted:
        if dims == 3:
            if phiSlice is None:
                field = field[:, 0, :]
            else:
                sliceIdx = max(min(phiSlice, phiMaxIdx), 0)
                field = field[:, sliceIdx, :]

        # For the 4D case, only a 2D projection can be plotted:
        if dims == 4:
            if phiSlice is None:
                phiSlicee = 0
            else:
                phiSlicee = max(min(phiSlice, phiMaxIdx), 0)

            if timeSlice is None:
                timeSlicee = 0
            else:
                timeSlicee = max(min(timeSlice, timeMaxIdx), 0)
            field = field[:, phiSlicee, :, timeSlicee]

        # Plotting the field using a filled contour.
        extent = [Rplot.min(), Rplot.max(), Zplot.min(), Zplot.max()]
        fieldPlotHndl = ax.imshow(field.T, origin='lower', cmap=ccmap,
                                  aspect='equal', extent=extent)

        # Selecting the name to display in the colorbar.

        ccbarname = {
            'br': '$B_r$ [T]',
            'bz': '$B_z$[T]',
            'bphi': '$B_\\phi$ [T]',
            'er': '$E_r$ [V/m]',
            'ez': '$E_z$ [V/m]',
            'ephi': '$E_\\phi$ [V/m]',
            'b': 'B [T]',
            'e': 'E [V/m]',
            'psipol': '$\\Psi_{pol}$ [Wb]'
            }.get(fieldName.lower())

        cbar = plt.colorbar(fieldPlotHndl, format=cbar_tick_format)
        cbar.set_label(ccbarname)

        # Preparing the axis labels:
        ax_options['xlabel'] = 'R [m]'
        ax_options['ylabel'] = 'Z [m]'
        ax = ssplt.axis_beauty(ax, ax_options)

        # Plotting the 2D vessel structures.
        if plot_vessel:
            try:
                ssplt.plot_vessel(linewidth=1, ax=ax)
            except AttributeError:
                pass
        plt.tight_layout()

        return ax

    def toDataset(self):
        """
        Export fields to a data set

        Useful to be compatible with the gyrokinetic object from iHIBPtracker
        WIP
        """
        data = xr.Dataset()
        if self.bdims == 2:
            data['Br'] = xr.DataArray(self.Bfield['fr'],
                                           dims=('R', 'z'),
                                           coords={'R':self.Bfield['R'],
                                        'z':self.Bfield['z']})
            data['Bz'] = xr.DataArray(self.Bfield['fz'],
                                dims=('R', 'z'),
                                coords={'R':self.Bfield['R'],
                                        'z':self.Bfield['z']})
            data['Bphi'] = xr.DataArray(self.Bfield['ft'],
                                dims=('R', 'z'),
                                coords={'R':self.Bfield['R'],
                                        'z':self.Bfield['z']})
            data['Psi'] = xr.DataArray(self.psipol['f'],
                                dims=('R', 'z'),
                                coords={'R':self.Bfield['R'],
                                        'z':self.Bfield['z']})
            data['B'] = np.sqrt(data['Br']**2 + data['Bz']**2+data['Bphi']**2)
        return data
                                        
