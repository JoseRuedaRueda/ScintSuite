"""Routines to read and write the fields for PSFT simulation codes"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import Lib.LibData as ssdat
import Lib._Plotting as ssplt
import Lib.errors as errors
import math
from Lib._Machine import machine


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
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False
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

    def readFiles(self, Bfile: str = None, Efile: str = None):
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
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False

        if Bfile is not None:
            with open(Bfile, 'rb') as fid:
                self.Bfield['nR'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nz'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nPhi'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nTime'] = np.fromfile(fid, 'uint32', 1)

                self.Bfield['Rmin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Rmax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['zmin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['zmax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Phimin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Phimax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Timemin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Timemax'] = np.fromfile(fid, 'float64', 1)

                size2read = self.Bfield['nR'] * self.Bfield['nz'] \
                    * self.Bfield['nPhi'] * self.Bfield['nTime']

                br = np.fromfile(fid, 'float64', count=size2read[0])
                bphi = np.fromfile(fid, 'float64', count=size2read[0])
                bz = np.fromfile(fid, 'float64', count=size2read[0])

                if self.Bfield['nR'] == 1:  # Just one point
                    self.bdims = 0
                    self.Bfield['fr'] = br.flatten()
                    self.Bfield['fz'] = bz.flatten()
                    self.Bfield['ft'] = bphi.flatten()

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['zmin'][0],
                                                   self.Bfield['zmax'][0],
                                                   self.Bfield['nz'][0])

                    self.Brinterp = lambda r, z, phi, time: self.Bfield['fr']
                    self.Bzinterp = lambda r, z, phi, time: self.Bfield['fz']
                    self.Bphiinterp = lambda r, z, phi, time: self.Bfield['ft']

                elif self.Bfield['nPhi'] == 1 and self.Bfield['nTime'] == 1:
                    self.bdims = 2  # Static 2D fields
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nz'][0]),
                                                   order='F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nz'][0]),
                                                   order='F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nz'][0]),
                                                     order='F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['zmin'][0],
                                                   self.Bfield['zmax'][0],
                                                   self.Bfield['nz'][0])

                    self.Brinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                self.Bfield['fr'], (r.flatten(), z.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                self.Bfield['fz'], (r.flatten(), z.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                self.Bfield['ft'], (r.flatten(), z.flatten()))
                elif self.Bfield['nTime'] == 1:
                    self.bdims = 3  # Static 3D fields
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nz'][0]),
                                                   order='F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nz'][0]),
                                                   order='F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nPhi'][0],
                                                      self.Bfield['nz'][0]),
                                                     order='F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['zmin'][0],
                                                   self.Bfield['zmax'][0],
                                                   self.Bfield['nz'][0])
                    self.Bfield['Phi'] = np.linspace(self.Bfield['Phimin'][0],
                                                     self.Bfield['Phimax'][0],
                                                     self.Bfield['nPhi'][0])

                    self.Brinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z']), self.Bfield['fr'],
                                (r.flatten(), phi.flatten(), z.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z']), self.Bfield['fz'],
                                (r.flatten(), phi.flatten(), z.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z']), self.Bfield['ft'],
                                (r.flatten(), phi.flatten(), z.flatten()))
                else:
                    self.bdims = 4  # Full 4D field
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nz'][0],
                                                    self.Bfield['nTime'][0]),
                                                   order='F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nz'][0],
                                                    self.Bfield['nTime'][0]),
                                                   order='F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nPhi'][0],
                                                      self.Bfield['nz'][0],
                                                      self.Bfield['nTime'][0]),
                                                     order='F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['zmin'][0],
                                                   self.Bfield['zmax'][0],
                                                   self.Bfield['nz'][0])
                    self.Bfield['Phi'] = np.linspace(self.Bfield['Phimin'][0],
                                                     self.Bfield['Phimax'][0],
                                                     self.Bfield['nPhi'][0])
                    self.Bfield['time'] = \
                        np.linspace(self.Bfield['Timemin'][0],
                                    self.Bfield['Timemax'][0],
                                    self.Bfield['nTime'][0])

                    self.Brinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z'], self.Bfield['time']),
                                self.Bfield['fr'],
                                (r.flatten(), phi.flatten(),
                                 z.flatten(), time.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z'], self.Bfield['time']),
                                self.Bfield['fz'],
                                (r.flatten(), phi.flatten(),
                                 z.flatten(), time.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z'], self.Bfield['time']),
                                self.Bfield['ft'],
                                (r.flatten(), phi.flatten(),
                                 z.flatten(), time.flatten()))
                del br
                del bz
                del bphi
        if Efile is not None:
            with open(Efile, 'rb') as fid:
                self.Efield['nR'] = np.fromfile(fid, 'uint32', 1)
                self.Efield['nz'] = np.fromfile(fid, 'uint32', 1)
                self.Efield['nPhi'] = np.fromfile(fid, 'uint32', 1)
                self.Efield['nTime'] = np.fromfile(fid, 'uint32', 1)

                self.Efield['Rmin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Rmax'] = np.fromfile(fid, 'float64', 1)
                self.Efield['zmin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['zmax'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Phimin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Phimax'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Timemin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Timemax'] = np.fromfile(fid, 'float64', 1)

                size2read = self.Efield['nR'] * self.Efield['nz']\
                    * self.Efield['nPhi'] * self.Efield['nTime']

                er = np.fromfile(fid, 'float64', size2read[0])
                ephi = np.fromfile(fid, 'float64', size2read[0])
                ez = np.fromfile(fid, 'float64', size2read[0])

                if self.Efield['nPhi'] == 1 and self.Efield['nTime'] == 1:
                    self.edims = 2
                    self.Efield['fr'] = er.reshape((self.Efield['nR'][0],
                                                    self.Efield['nz'][0]),
                                                   order='F')
                    self.Efield['fz'] = ez.reshape((self.Efield['nR'][0],
                                                    self.Efield['nz'][0]),
                                                   order='F')
                    self.Efield['ft'] = ephi.reshape((self.Efield['nR'][0],
                                                      self.Efield['nz'][0]),
                                                     order='F')

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'][0],
                                                   self.Efield['Rmax'][0],
                                                   self.Efield['nR'][0])
                    self.Efield['z'] = np.linspace(self.Efield['zmin'][0],
                                                   self.Efield['zmax'][0],
                                                   self.Efield['nz'][0])

                    self.Erinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z']),
                                self.Efield['fr'], (r.flatten(), z.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z']),
                                self.Efield['fz'], (r.flatten(), z.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z']),
                                self.Efield['ft'], (r.flatten(), z.flatten()))
                elif self.Efield['nTime'] == 1:
                    self.edims = 3
                    self.Efield['fr'] = er.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nz'][0]),
                                                   order='F')
                    self.Efield['fz'] = ez.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nz'][0]),
                                                   order='F')
                    self.Efield['ft'] = ephi.reshape((self.Efield['nR'][0],
                                                      self.Efield['nPhi'][0],
                                                      self.Efield['nz'][0]),
                                                     order='F')

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'][0],
                                                   self.Efield['Rmax'][0],
                                                   self.Efield['nR'][0])
                    self.Efield['z'] = np.linspace(self.Efield['zmin'][0],
                                                   self.Efield['zmax'][0],
                                                   self.Efield['nz'][0])
                    self.Efield['phi'] = np.linspace(self.Efield['Phimin'][0],
                                                     self.Efield['Phimax'][0],
                                                     self.Efield['nPhi'][0])

                    self.Erinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z'],
                                 self.Efield['Phi']), self.Efield['fr'],
                                (r.flatten(), z.flatten(), phi.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z'],
                                 self.Efield['Phi']), self.Efield['fz'],
                                (r.flatten(), z.flatten(), phi.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z'],
                                 self.Efield['Phi']), self.Efield['ft'],
                                (r.flatten(), z.flatten(), phi.flatten()))
                else:
                    self.edims = 4
                    self.Efield['fr'] = er.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nz'][0],
                                                    self.Efield['nTime'][0]),
                                                   order='F')
                    self.Efield['fz'] = ez.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nz'][0],
                                                    self.Efield['nTime'][0]),
                                                   order='F')
                    self.Efield['ft'] = ephi.reshape((self.Efield['nR'][0],
                                                      self.Efield['nPhi'][0],
                                                      self.Efield['nz'][0],
                                                      self.Efield['nTime'][0]),
                                                     order='F')

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'][0],
                                                   self.Efield['Rmax'][0],
                                                   self.Efield['nR'][0])
                    self.Efield['z'] = np.linspace(self.Efield['zmin'][0],
                                                   self.Efield['zmax'][0],
                                                   self.Efield['nz'][0])
                    self.Efield['Phi'] = np.linspace(self.Efield['Phimin'][0],
                                                     self.Efield['Phimax'][0],
                                                     self.Efield['nPhi'][0])
                    self.Efield['time'] = \
                        np.linspace(self.Efield['Timemin'][0],
                                    self.Efield['Timemax'][0],
                                    self.Efield['nTime'][0])

                    self.Erinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z'],
                                 self.Efield['Phi'], self.Efield['time']),
                                self.Efield['fr'],
                                (r.flatten(), z.flatten(),
                                 phi.flatten(), time.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z'],
                                 self.Efield['Phi'], self.Efield['time']),
                                self.Efield['fz'],
                                (r.flatten(), z.flatten(),
                                 phi.flatten(), time.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                        interpn((self.Efield['R'], self.Efield['z'],
                                 self.Efield['Phi'], self.Efield['time']),
                                self.Efield['ft'],
                                (r.flatten(), z.flatten(),
                                 phi.flatten(), time.flatten()))

                del er
                del ez
                del ephi

    def readBfromDB(self, shotnumber: int = 34570, time: float = 2.5,
                    exp: str = 'AUGD', diag: str = 'EQI',
                    edition: int = 0,
                    Rmin: float = 1.03, Rmax: float = 2.65,
                    zmin: float = -1.224, zmax: float = 1.05,
                    nR: int = 128, nz: int = 256):
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
                                             time=time,
                                             exp=exp,
                                             ed=edition,
                                             diag=diag)
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
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        if machine != 'MU':
            # Retrieving as well the poloidal magnetic flux.
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
                               time: float = 2.5, R0: float = 190.0,
                               z0: float = 92.0,
                               exp: str = 'AUGD', diag: str = 'EQI',
                               edition: int = 0,
                               Rmin: float = 160.0, Rmax: float = 220.0,
                               zmin: float = 80.0, zmax: float = 120.0,
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
        """
        self.bdims = 0

        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR) / 100.0
        z = np.linspace(zmin, zmax, num=nz) / 100.0
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
                         nR: int = 128, nz: int = 256):
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
        """
        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        psipol = ssdat.get_psipol(shotnumber, RR.flatten(), zz.flatten(),
                                  diag=diag, time=time)

        # Reshaping into the original shape.
        psipol = np.reshape(psipol, grid_shape).T

        # Creating the interpolating function.
        self.psipol['R'] = np.array(R, dtype=np.float64)
        self.psipol['z'] = np.array(z, dtype=np.float64)
        self.psipol['nR'] = np.array([nR], dtype=np.int32)
        self.psipol['nz'] = np.array([nz], dtype=np.int32)
        self.psipol['f'] = psipol.astype(dtype=np.float64)
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
        """
        if self.psipol_on:
            psipol = self.psipol_interp(R, z, phi, t)
        else:
            raise Exception('The poloidal flux is not loaded!')

        return psipol

    def tofile(self, fid, bflag: bool = True, eflag: bool = False):
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
        """
        if isinstance(fid, str):
            fid = open(fid, 'wb')
            opened = True
        else:
            opened = False
        if bflag is False and eflag is False:
            raise Exception('Some flag has to be set to write to file!')
        if bflag:
            # Write header with grid size information:
            self.Bfield['nR'].tofile(fid)
            self.Bfield['nz'].tofile(fid)
            self.Bfield['nPhi'].tofile(fid)
            self.Bfield['nTime'].tofile(fid)

            # Write grid ends:
            self.Bfield['Rmin'].tofile(fid)
            self.Bfield['Rmax'].tofile(fid)
            self.Bfield['zmin'].tofile(fid)
            self.Bfield['zmax'].tofile(fid)
            self.Bfield['Phimin'].tofile(fid)
            self.Bfield['Phimax'].tofile(fid)
            self.Bfield['Timemin'].tofile(fid)
            self.Bfield['Timemax'].tofile(fid)

            # Write fields
            if self.bdims > 0:
                self.Bfield['fr'].ravel(order='F').tofile(fid)
                self.Bfield['ft'].ravel(order='F').tofile(fid)
                self.Bfield['fz'].ravel(order='F').tofile(fid)
            else:
                self.Bfield['fx'].ravel(order='F').tofile(fid)
                self.Bfield['fy'].ravel(order='F').tofile(fid)
                self.Bfield['fz'].ravel(order='F').tofile(fid)
        elif eflag:
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

        else:
            raise Exception('Not a valid combination of inputs')
        if opened:
            fid.close()

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
        fieldPlotHndl = ax.contourf(Rplot, Zplot, field.T, nLevels, cmap=ccmap)

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
            ssplt.plot_vessel(linewidth=1, ax=ax)
        plt.tight_layout()

        return ax
