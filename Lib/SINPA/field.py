"""
Library to prepare and plot the input magnetic field to SINPA

Jose Rueda: jrrueda@us.es

Introduced in version 0.5.9
"""
import os
import f90nml
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import Lib.LibData as ssdat
import Lib.LibPlotting as ssplt
from Lib.LibPaths import Path
from Lib.LibMachine import machine
paths = Path(machine)


# -----------------------------------------------------------------------------
# --- Angles and orientation
# -----------------------------------------------------------------------------
def calculate_sinpa_angles(B: np.ndarray, geomID: str = 'Test'):
    """
    Calculate the zita and epsilon angles for SINPA

    Jose Rueda: jrrueda@us.es

    @param B: Magnetic field vector (full or just unit vector)
    @param geomID: ID of the geometry to be used

    @return zita: zita angle [deg]
    @return ipsilon: ipsilon angle [deg]
    """
    # --- Load the vectors
    filename = os.path.join(paths.SINPA, 'Geometry', geomID,
                            'ExtraGeometryParams.txt')
    nml = f90nml.read(filename)
    u1 = np.array(nml['ExtraGeometryParams']['u1'])
    u2 = np.array(nml['ExtraGeometryParams']['u2'])
    u3 = np.array(nml['ExtraGeometryParams']['u3'])

    # --- Calculate the zeta angle
    bmod = math.sqrt(np.sum(B * B))
    zita = math.acos(np.sum(B * u3) / bmod) * 180. / math.pi

    # --- Calculate the ipsilon angle
    ipsilon = math.atan2(np.sum(B * u2), np.sum(B * u1)) * 180. / math.pi
    return zita, ipsilon


def constructDirection(zita, ipsilon, geomID: str = 'Test'):
    """
    Calculate the zita and epsilon angles for SINPA

    Jose Rueda: jrrueda@us.es

    @param zita: zita angle [deg]
    @param ipsilon: ipsilon angle [deg]
    @param geomID: ID of the geometry to be used

    @return B: director vector
    """
    # --- Load the vectors
    filename = os.path.join(paths.SINPA, 'Geometry', geomID,
                            'ExtraGeometryParams.txt')
    nml = f90nml.read(filename)
    u1 = np.array(nml['ExtraGeometryParams']['u1'])
    u2 = np.array(nml['ExtraGeometryParams']['u2'])
    u3 = np.array(nml['ExtraGeometryParams']['u3'])

    direction = (math.cos(ipsilon * math.pi / 180.0) * u1
                 + math.sin(ipsilon * math.pi / 180.0) * u2) \
        * math.sin(zita * math.pi / 180.0) \
        + math.cos(zita * math.pi / 180.0) * u3
    return direction


# -----------------------------------------------------------------------------
# --- SINPA field class
# -----------------------------------------------------------------------------
class sinpaField:
    """Class with the SINPA fields"""

    def __init__(self):
        """
        Initialize a dummy object.

        Call the class methods:
        a) readFiles: to read from the SINPA file
        b) readBfromAUG: to fetch the fields from the AUG database.

        Pablo Oyola - pablo.oyola@ipp.mpg.de ft. Jose Rueda: jrrueda@us.es
        """
        self.bdims = 0
        self.Bfield = {'R': np.array((0), dtype=np.float64),
                       'z': np.array((0), dtype=np.float64),
                       'fr': np.array((0), dtype=np.float64),
                       'fz': np.array((0), dtype=np.float64),
                       'ft': np.array((0), dtype=np.float64),
                       'nPhi': np.array((1), dtype=np.int32),
                       'Phimin': np.array((0.0), dtype=np.float64),
                       'Phimax': np.array((2.0*np.pi), dtype=np.float64),
                       }

        self.Bfield_from_shot_flag = False

    def readFiles(self, Bfile: str = None):
        """
        Start the class containing the E-M fields from files

        Pablo Oyola ft. Jose Rueda
        Note that its follow the i-HIBPsim/SINPA structure
        (SINPA fields are similar to iHIBPsim fields
        but with no time dependence):
        --> nR, nZ, nPhi: int32 - Grid size in each direction.
        --> Rmin, Rmax: float64 - Minimum and maximum major radius.
        --> Zmin, Zmax: float64 - Minimum and maximum vertical pos.
        --> Phimin, Phimax: float64 - Min. and max. toroidal direction.
        --> Br[nR, nPhi, nZ]: float64
        --> Bphi[nR, nPhi, nZ]: float64
        --> Bz[nR, nPhi, nZ]: float64

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param Bfile: Full path to the magnetic field.
        """
        self.bdims = 0

        if Bfile is not None:
            with open(Bfile, 'rb') as fid:
                self.Bfield['nR'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nZ'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nPhi'] = np.fromfile(fid, 'uint32', 1)

                self.Bfield['Rmin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Rmax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Zmin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Zmax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Phimin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Phimax'] = np.fromfile(fid, 'float64', 1)

                size2read = self.Bfield['nR'] * self.Bfield['nZ'] \
                    * self.Bfield['nPhi']

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
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'][0],
                                                   self.Bfield['Zmax'][0],
                                                   self.Bfield['nZ'][0])

                    self.Brinterp = lambda r, z, phi: self.Bfield['fr']
                    self.Bzinterp = lambda r, z, phi: self.Bfield['fz']
                    self.Bphiinterp = lambda r, z, phi: self.Bfield['ft']
                elif self.Bfield['nPhi'] == 1:  # 2D field
                    self.bdims = 2
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nZ'][0]),
                                                   order='F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nZ'][0]),
                                                   order='F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nZ'][0]),
                                                     order='F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'][0],
                                                   self.Bfield['Zmax'][0],
                                                   self.Bfield['nZ'][0])

                    self.Brinterp = lambda r, z, phi: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                self.Bfield['fr'], (r.flatten(), z.flatten()))
                    self.Bzinterp = lambda r, z, phi: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                self.Bfield['fz'], (r.flatten(), z.flatten()))
                    self.Bphiinterp = lambda r, z, phi: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                self.Bfield['ft'], (r.flatten(), z.flatten()))
                else:  # Full 3D field
                    self.bdims = 3
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nZ'][0]),
                                                   order='F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nZ'][0]),
                                                   order='F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nPhi'][0],
                                                      self.Bfield['nZ'][0]),
                                                     order='F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'][0],
                                                   self.Bfield['Zmax'][0],
                                                   self.Bfield['nZ'][0])
                    self.Bfield['Phi'] = np.linspace(self.Bfield['Phimin'][0],
                                                     self.Bfield['Phimax'][0],
                                                     self.Bfield['nPhi'][0])

                    self.Brinterp = lambda r, z, phi: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z']), self.Bfield['fr'],
                                (r.flatten(), phi.flatten(), z.flatten()))
                    self.Bzinterp = lambda r, z, phi: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z']), self.Bfield['fz'],
                                (r.flatten(), phi.flatten(), z.flatten()))
                    self.Bphiinterp = lambda r, z, phi: \
                        interpn((self.Bfield['R'], self.Bfield['Phi'],
                                 self.Bfield['z']), self.Bfield['ft'],
                                (r.flatten(), phi.flatten(), z.flatten()))
                # Clean memory
                del br
                del bz
                del bphi

    def readBfromDB(self, shotnumber: int = 39612, time: float = 2.5,
                    exp: str = 'AUGD', diag: str = 'EQI',
                    edition: int = 0,
                    Rmin: float = 160.0, Rmax: float = 220.0,
                    zmin: float = 80.0, zmax: float = 120.0,
                    nR: int = 40, nZ: int = 80):
        """
        Read field from machine database

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        ft.
        Jose Rueda: jrrueda@us.es

        @param shotnumber: Shot from which to extract the magnetic equilibrium.
        @param time: Time point to fetch the equilibrium.
        @param exp: Experiment where the equilibria is stored.
        @param diag: Diagnostic from which extracting the equilibrium.
        @param edition: Edition of the equilibrium to retrieve. Set to 0 by
        default, which will take from the AUG DB the latest version.
        @param Rmin: Minimum radius to get the magnetic equilibrium.
        @param Rmax: Maximum radius to get the magnetic equilibrium.
        @param zmin: Minimum Z to get the magnetic equilibrium.
        @param zmax: Maximum Z to get the magnetic equilibrium.
        @param nR: Number of points to define the B field grid in R direction.
        @param nZ: Number of points to define the B field grid in Z direction.
        """
        self.bdims = 0

        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR) / 100.0
        z = np.linspace(zmin, zmax, num=nZ) / 100.0
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
        self.Bfield['Zmin'] = np.array((zmin), dtype=np.float64)
        self.Bfield['Zmax'] = np.array((zmax), dtype=np.float64)
        self.Bfield['nR'] = np.array([nR], dtype=np.int32)
        self.Bfield['nZ'] = np.array([nZ], dtype=np.int32)
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
        self.Bfield_from_shot_flag = True
        self.shotnumber = shotnumber
        self.edition = edition
        self.timepoint = time
        self.diag = diag
        self.exp = exp

    def readBfromDBSinglePoint(self, shotnumber: int = 39612,
                               time: float = 2.5, R0: float = 190.0,
                               z0: float = 92.0,
                               exp: str = 'AUGD', diag: str = 'EQI',
                               edition: int = 0,
                               Rmin: float = 160.0, Rmax: float = 220.0,
                               zmin: float = 80.0, zmax: float = 120.0,
                               nR: int = 40, nZ: int = 80):
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

        @param shotnumber: Shot from which to extract the magnetic equilibrium.
        @param time: Time point to fetch the equilibrium.
        @param R0: R where to calculate the magnetic field [m]
        @param Z0: Z where to calculate the
        @param exp: Experiment where the equilibria is stored.
        @param diag: Diagnostic from which extracting the equilibrium.
        @param edition: Edition of the equilibrium to retrieve. Set to 0 by
        default, which will take from the AUG DB the latest version.
        @param Rmin: Minimum radius to get the magnetic equilibrium.
        @param Rmax: Maximum radius to get the magnetic equilibrium.
        @param zmin: Minimum Z to get the magnetic equilibrium.
        @param zmax: Maximum Z to get the magnetic equilibrium.
        @param nR: Number of points to define the B field grid in R direction.
        @param nZ: Number of points to define the B field grid in Z direction.
        """
        self.bdims = 0

        # Getting from the database.
        R = np.linspace(Rmin, Rmax, num=nR) / 100.0
        z = np.linspace(zmin, zmax, num=nZ) / 100.0
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
        self.Bfield['Zmin'] = np.array((zmin), dtype=np.float64)
        self.Bfield['Zmax'] = np.array((zmax), dtype=np.float64)
        self.Bfield['nR'] = np.array([nR], dtype=np.int32)
        self.Bfield['nZ'] = np.array([nZ], dtype=np.int32)
        self.Bfield['fr'] = Br.astype(dtype=np.float64)
        self.Bfield['fz'] = Bz.astype(dtype=np.float64)
        self.Bfield['ft'] = Bt.astype(dtype=np.float64)

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        # Calculate the value of the field in the desired point
        rr0 = np.array(R0)
        zz0 = np.array(z0)
        bbr = self.Brinterp(rr0, zz0, 0.0)
        bbz = self.Bzinterp(rr0, zz0, 0.0)
        bbphi = self.Bphiinterp(rr0, zz0, 0.0)

        # Impose this value in all the grid points:
        self.Bfield['fr'][:] = bbr
        self.Bfield['fz'][:] = bbz
        self.Bfield['ft'][:] = bbphi

        # Re-create the interpolators
        self.Brinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        # Saving the input data to the class.
        self.Bfield_from_shot_flag = True
        self.shotnumber = shotnumber
        self.edition = edition
        self.timepoint = time
        self.diag = diag
        self.exp = exp

    def createFromSingleB(self, B: np.ndarray, Rmin: float = 160.0,
                          Rmax: float = 220.0,
                          zmin: float = 80.0, zmax: float = 120.0,
                          nR: int = 40, nZ: int = 80):
        """
        Create a field for SINPA from a given B vector

        Jose Rueda: jrrueda@us.es

        The idea is that it will take the given magnetic field and
        then use this value in all points in the grid. Notice that we will not
        create an uniform grid in this way because we will be putting the same
        2D grid, so we will be creating a 'toroidally homogeneous field' (if
        this term exist).

        @param B: Magnetic field to be used [Br,Bz,Bphi] [T]
        @param Rmin: Minimum radius to get the magnetic equilibrium.
        @param Rmax: Maximum radius to get the magnetic equilibrium.
        @param zmin: Minimum Z to get the magnetic equilibrium.
        @param zmax: Maximum Z to get the magnetic equilibrium.
        @param nR: Number of points to define the B field grid in R direction.
        @param nZ: Number of points to define the B field grid in Z direction.
        """
        self.bdims = 0

        # Prepare the grid and allocate the field
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nZ)
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
        self.Bfield['Zmin'] = np.array((zmin), dtype=np.float64)
        self.Bfield['Zmax'] = np.array((zmax), dtype=np.float64)
        self.Bfield['nR'] = np.array([nR], dtype=np.int32)
        self.Bfield['nZ'] = np.array([nZ], dtype=np.int32)
        self.Bfield['fr'] = Br.astype(dtype=np.float64)
        self.Bfield['fz'] = Bz.astype(dtype=np.float64)
        self.Bfield['ft'] = Bt.astype(dtype=np.float64)

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fr'],
                    (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['fz'],
                    (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi: \
            interpn((self.Bfield['R'], self.Bfield['z']), self.Bfield['ft'],
                    (r.flatten(), z.flatten()))

        # Saving the input data to the class.
        self.Bfield_from_shot_flag = False

    def getBfield(self, R: float, z: float,
                  phi: float = None):
        """
        Get the magnetic field components at the given points.

        Note, it extract the field using the interpolators

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param R: Major radius to evaluate the magnetic field.
        @param z: Vertical position to evaluate the magnetic field.
        @param phi: Toroidal location to evaluate the magnetic field.
        If the system is in 2D, it will be ignored.

        @return Br: Radial component of the magnetic field.
        @return Bz: Vertical component of the magnetic field.
        @return Bphi: Toroidal component of the magnetic field.

        """
        if self.bdims != 0:
            Br = self.Brinterp(R, z, phi)
            Bz = self.Bzinterp(R, z, phi)
            Bphi = self.Bphiinterp(R, z, phi)
        else:
            raise Exception('The magnetic field has not been loaded!')

        return Br, Bz, Bphi

    def tofile(self, fid):
        """
        Write the magnetic field to files following the SINPA scheme.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param fid: file identifier where the files will be written.
        @param bflag: states if the magnetic field has to be written.
        Default is True, so magnetic field will be written.
        @param eflag: states if the electric field has to be written.
        Default to False. If this is set to True, the magnetic field
        will not be written.
        """
        # Write header with grid size information:
        self.Bfield['nR'].tofile(fid)
        self.Bfield['nZ'].tofile(fid)
        self.Bfield['nPhi'].tofile(fid)

        # Write grid ends:
        self.Bfield['Rmin'].tofile(fid)
        self.Bfield['Rmax'].tofile(fid)
        self.Bfield['Zmin'].tofile(fid)
        self.Bfield['Zmax'].tofile(fid)
        self.Bfield['Phimin'].tofile(fid)
        self.Bfield['Phimax'].tofile(fid)

        # Write fields
        # ToDo: check that this works with a full 3D field! (I am not sure
        # about the T for a 3D array)
        self.Bfield['fr'].T.tofile(fid)
        self.Bfield['ft'].T.tofile(fid)
        self.Bfield['fz'].T.tofile(fid)

    def plot(self, fieldName: str = 'bphi', phiSlice: int = None,
             ax_options: dict = {}, ax=None, cmap=None, nLevels: int = 50,
             cbar_tick_format: str = '%.2e', plot_vessel: bool = True):
        """
        Plot the input field

        Plots the input field ('Br', 'Bz', 'Bphi','B',) into some axis, ax,
        or the routine creates one for the plotting.

        @param fieldName: Name of the field to be plotted.
        @param ax_options: options for the function axis_beauty.
        @param ax: Axis to plot the data.
        @param cmap: Colormap to use. Gamma_II is set by default.
        @param ax: Return the axis where the plot has been done.
        @param plot_vessel: Flag to plot the vessel
        """
        # --- Initialise the plotting parameters
        ax_options['ratio'] = 'equal'   # The ratio must be always equal

        if 'grid' not in ax_options:
            ax_options['grid'] = 'both'

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
        if fieldName[0] == 'b':
            if self.bdims == 0:
                raise Exception('Plot not working of 0D fields')
            dims = self.bdims
            if dims == 3:
                phiMaxIdx = self.Bfield['nPhi']

        # Get the appropriate field
        field = {
            'br': self.Bfield['fr'],
            'bz': self.Bfield['fz'],
            'bphi': self.Bfield['ft'],
            'b': np.sqrt(self.Bfield['fr']**2
                         + self.Bfield['fz']**2
                         + self.Bfield['ft']**2),
        }.get(fieldName)

        Rplot = {
            'br': self.Bfield['R'],
            'bz': self.Bfield['R'],
            'bphi': self.Bfield['R'],
            'b': self.Bfield['R'],
        }.get(fieldName)

        Zplot = {
            'br': self.Bfield['z'],
            'bz': self.Bfield['z'],
            'bphi': self.Bfield['z'],
            'b': self.Bfield['z'],
        }.get(fieldName)

        # For the 3D case, only a 2D projection can be plotted:
        if dims == 3:
            if phiSlice is None:
                field = field[:, 0, :]
            else:
                sliceIdx = max(min(phiSlice, phiMaxIdx), 0)
                field = field[:, sliceIdx, :]

        # Plotting the field using a filled contour.
        fieldPlotHndl = ax.contourf(Rplot, Zplot, field.T, nLevels, cmap=ccmap)

        # Selecting the name to display in the colorbar.

        ccbarname = {
            'br': '$B_r$ [T]',
            'bz': '$B_z$[T]',
            'bphi': '$B_\\phi$ [T]',
            'b': 'B [T]',
            'psipol': '$\\Psi_{pol}$ [Wb]'
            }.get(fieldName)

        cbar = plt.colorbar(fieldPlotHndl, format=cbar_tick_format)
        cbar.set_label(ccbarname)

        # Preparing the axis labels:
        ax_options['xlabel'] = 'Major radius R [m]'
        ax_options['ylabel'] = 'z [m]'
        ax = ssplt.axis_beauty(ax, ax_options)

        # Plotting the 2D vessel structures.
        if plot_vessel:
            ssplt.plot_vessel(linewidth=1, ax=ax)
        plt.tight_layout()

        return ax
