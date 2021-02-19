"""Contains the routines to read and write the
fields and profiles from i-HIBPsim."""

import numpy as np
import warnings
import dd
#from version_suite import version
from LibMachine import machine
import LibPlotting as ssplt
#import LibParameters as sspar
import matplotlib.pyplot as plt
#from scipy.interpolate import RectBivariateSpline as intp2
from scipy.interpolate import interpn
if machine == 'AUG':
    import LibDataAUG as ssdat

class ihibpEMfields:
    def __init__(self):
        """
        Initializes a dummy object. Call the class methods:
        a) readFiles: to read from the files the EM fields.
        b) readBfromAUG: to fetch the fields from the AUG database.

        Pablo Oyola: pablo.oyola@ipp.mpg.de
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False
        self.Bfield = {'R': np.array((0), dtype = np.float64),
                       'z': np.array((0), dtype = np.float64),
                       'fr': np.array((0), dtype = np.float64), 
                       'fz': np.array((0), dtype = np.float64), 
                       'ft': np.array((0), dtype = np.float64),
                       'nPhi': np.array((1), dtype = np.int32),
                       'nTime': np.array((1), dtype = np.int32),
                       'Phimin': np.array((0.0), dtype = np.float64),
                       'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                       'Timemin': np.array((0.0), dtype = np.float64),
                       'Timemax': np.array((1.0), dtype = np.float64)
                       }
        
        self.Efield = {'R': np.array((0), dtype = np.float64),
                       'z': np.array((0), dtype = np.float64), 
                       'fr': np.array((0), dtype = np.float64), 
                       'fz': np.array((0), dtype = np.float64), 
                       'ft': np.array((0), dtype = np.float64),
                       'nPhi': np.array((1), dtype = np.int32),
                       'nTime': np.array((1), dtype = np.int32),
                       'Phimin': np.array((0.0), dtype = np.float64),
                       'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                       'Timemin': np.array((0.0), dtype = np.float64),
                       'Timemax': np.array((1.0), dtype = np.float64)
                       }
        
        self.psipol = {'R': np.array((0), dtype = np.float64),
                       'z': np.array((0), dtype = np.float64), 
                       'f': np.array((0), dtype = np.float64),
                       'nPhi': np.array((1), dtype = np.int32),
                       'nTime': np.array((1), dtype = np.int32),
                       'Phimin': np.array((0.0), dtype = np.float64),
                       'Phimax': np.array((2.0*np.pi), dtype = np.float64),
                       'Timemin': np.array((0.0), dtype = np.float64),
                       'Timemax': np.array((1.0), dtype = np.float64)
                       }
        
        self.Bfield_from_shot_flag = False

    def readFiles(self, Bfile: str = None, Efile: str = None):
        """
        Starts the class containing the E-M fields from files following the
        i-HIBPsim structure:
        --> nR, nZ, nPhi, nTime: int32 - Grid size in each direction.
        --> Rmin, Rmax: float64 - Minimum and maximum major radius.
        --> Zmin, Zmax: float64 - Minimum and maximum vertical pos.
        --> Phimin, Phimax: float64 - Min. and max. toroidal direction.
        --> Timemin, Timemax: float64 - Min. and max. times.
        --> Br[nR, nPhi, nZ, nTime]: float64
        --> Bphi[nR, nPhi, nZ, nTime]: float64
        --> Bz[nR, nPhi, nZ, nTime]: float64

        Pablo Oyola: pablo.oyola@ipp.mpg.de

        @param Bfile: Full path to the magnetic field.
        @param Efile: Full path to the electric field.
        """

        self.bdims = 0
        self.edims = 0
        self.psipol_on = False

        if Bfile is not None:
            with open(Bfile, 'rb') as fid:
                self.Bfield['nR'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nZ'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nPhi'] = np.fromfile(fid, 'uint32', 1)
                self.Bfield['nTime'] = np.fromfile(fid, 'uint32', 1)

                self.Bfield['Rmin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Rmax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Zmin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Zmax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Phimin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Phimax'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Timemin'] = np.fromfile(fid, 'float64', 1)
                self.Bfield['Timemax'] = np.fromfile(fid, 'float64', 1)

                size2read = self.Bfield['nR'] * self.Bfield['nZ'] *\
                            self.Bfield['nPhi'] * self.Bfield['nTime']

                br = np.fromfile(fid, 'float64', count=size2read[0])
                bphi = np.fromfile(fid, 'float64', count=size2read[0])
                bz = np.fromfile(fid, 'float64', count=size2read[0])

                if self.Bfield['nPhi'] == 1 and self.Bfield['nTime'] == 1:
                    self.bdims = 2
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nZ'][0]),
                                                    order = 'F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nZ'][0]),
                                                    order = 'F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nZ'][0]),
                                                    order = 'F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0], \
                                                   self.Bfield['Rmax'][0], \
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'][0], \
                                                   self.Bfield['Zmax'][0], \
                                                   self.Bfield['nZ'][0])

                    self.Brinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z']), \
                                            self.Bfield['fr'], \
                                            (r.flatten(), z.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z']), \
                                            self.Bfield['fz'], \
                                            (r.flatten(), z.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z']), \
                                            self.Bfield['ft'], \
                                            (r.flatten(), z.flatten()))
                elif self.Bfield['nTime'] == 1:
                    self.bdims = 3
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nZ'][0]),
                                                    order = 'F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nZ'][0]),
                                                    order = 'F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nPhi'][0],
                                                      self.Bfield['nZ'][0]),
                                                      order = 'F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0], \
                                                   self.Bfield['Rmax'][0], \
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'][0], \
                                                   self.Bfield['Zmax'][0], \
                                                   self.Bfield['nZ'][0])
                    self.Bfield['Phi'] = np.linspace(self.Bfield['Phimin'][0],
                                                     self.Bfield['Phimax'][0],
                                                     self.Bfield['nPhi'][0])

                    self.Brinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['Phi'], \
                                            self.Bfield['z']), \
                                            self.Bfield['fr'], \
                                            (r.flatten(), phi.flatten(), \
                                             z.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['Phi'], \
                                            self.Bfield['z']), \
                                            self.Bfield['fz'], \
                                            (r.flatten(), phi.flatten(), \
                                             z.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['Phi'], \
                                            self.Bfield['z']), \
                                            self.Bfield['ft'], \
                                            (r.flatten(), phi.flatten(), \
                                             z.flatten()))
                else:
                    self.bdims = 4
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nZ'][0],
                                                    self.Bfield['nTime'][0]),
                                                    order = 'F')
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'][0],
                                                    self.Bfield['nPhi'][0],
                                                    self.Bfield['nZ'][0],
                                                    self.Bfield['nTime'][0]),
                                                    order = 'F')
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'][0],
                                                      self.Bfield['nPhi'][0],
                                                      self.Bfield['nZ'][0],
                                                      self.Bfield['nTime'][0]),
                                                      order = 'F')

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'][0],
                                                   self.Bfield['Rmax'][0],
                                                   self.Bfield['nR'][0])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'][0],
                                                   self.Bfield['Zmax'][0],
                                                   self.Bfield['nZ'][0])
                    self.Bfield['Phi'] = np.linspace(self.Bfield['Phimin'][0],
                                                     self.Bfield['Phimax'][0],
                                                     self.Bfield['nPhi'][0])
                    self.Bfield['time'] = np.linspace(self.Bfield['Timemin'][0],
                                                      self.Bfield['Timemax'][0],
                                                      self.Bfield['nTime'][0])

                    self.Brinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'],
                                            self.Bfield['Phi'],
                                            self.Bfield['z'],
                                            self.Bfield['time']),
                                            self.Bfield['fr'],
                                            (r.flatten(), phi.flatten(),
                                             z.flatten(), time.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'],
                                            self.Bfield['Phi'],
                                            self.Bfield['z'],
                                            self.Bfield['time']),
                                            self.Bfield['fz'],
                                            (r.flatten(), phi.flatten(),
                                             z.flatten(), time.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'],
                                            self.Bfield['z'],
                                            self.Bfield['Phi'],
                                            self.Bfield['time']),
                                            self.Bfield['ft'],
                                            (r.flatten(), phi.flatten(),
                                             z.flatten(), time.flatten()))
                
                del br
                del bz
                del bphi
        if Efile is not None:
            with open(Efile, 'rb') as fid:
                self.Efield['nR'] = np.fromfile(fid, 'uint32', 1)
                self.Efield['nZ'] = np.fromfile(fid, 'uint32', 1)
                self.Efield['nPhi'] = np.fromfile(fid, 'uint32', 1)
                self.Efield['nTime'] = np.fromfile(fid, 'uint32', 1)

                self.Efield['Rmin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Rmax'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Zmin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Zmax'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Phimin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Phimax'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Timemin'] = np.fromfile(fid, 'float64', 1)
                self.Efield['Timemax'] = np.fromfile(fid, 'float64', 1)

                size2read = self.Efield['nR'] * self.Efield['nZ'] *\
                            self.Efield['nPhi'] * self.Efield['nTime']

                er = np.fromfile(fid, 'float64', size2read[0])
                ephi = np.fromfile(fid, 'float64', size2read[0])
                ez = np.fromfile(fid, 'float64', size2read[0])

                if self.Efield['nPhi'] == 1 and self.Efield['nTime'] == 1:
                    self.edims = 2
                    self.Efield['fr'] = er.reshape((self.Efield['nR'][0],
                                                    self.Efield['nZ'][0]),
                                                   order = 'F')
                    self.Efield['fz'] = ez.reshape((self.Efield['nR'][0],
                                                    self.Efield['nZ'][0]),
                                                    order = 'F')
                    self.Efield['ft'] = ephi.reshape((self.Efield['nR'][0],
                                                      self.Efield['nZ'][0]),
                                                      order = 'F')

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'][0],
                                                   self.Efield['Rmax'][0],
                                                   self.Efield['nR'][0])
                    self.Efield['z'] = np.linspace(self.Efield['Zmin'][0],
                                                   self.Efield['Zmax'][0],
                                                   self.Efield['nZ'][0])

                    self.Erinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z']),
                                            self.Efield['fr'],
                                            (r.flatten(), z.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z']),
                                            self.Efield['fz'],
                                            (r.flatten(), z.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z']),
                                            self.Efield['ft'],
                                            (r.flatten(), z.flatten()))
                elif self.Efield['nTime'] == 1:
                    self.edims = 3
                    self.Efield['fr'] = er.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nZ'][0]),
                                                    order = 'F')
                    self.Efield['fz'] = ez.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nZ'][0]),
                                                    order = 'F')
                    self.Efield['ft'] = ephi.reshape((self.Efield['nR'][0],
                                                      self.Efield['nPhi'][0],
                                                      self.Efield['nZ'][0]),
                                                      order = 'F')
                    
                    self.Efield['R'] = np.linspace(self.Efield['Rmin'][0],
                                                   self.Efield['Rmax'][0],
                                                   self.Efield['nR'][0])
                    self.Efield['z'] = np.linspace(self.Efield['Zmin'][0],
                                                   self.Efield['Zmax'][0],
                                                   self.Efield['nZ'][0])
                    self.Efield['phi'] = np.linspace(self.Efield['Phimin'][0],
                                                     self.Efield['Phimax'][0],
                                                     self.Efield['nPhi'][0])
                    
                    self.Erinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z'],
                                            self.Efield['Phi']),
                                            self.Efield['fr'],
                                            (r.flatten(), z.flatten(),
                                             phi.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z'],
                                            self.Efield['Phi']),
                                            self.Efield['fz'],
                                            (r.flatten(), z.flatten(),
                                             phi.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z'],
                                            self.Efield['Phi']),
                                            self.Efield['ft'],
                                            (r.flatten(), z.flatten(),
                                             phi.flatten()))
                else:
                    self.edims = 4
                    self.Efield['fr'] = er.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nZ'][0],
                                                    self.Efield['nTime'][0]),
                                                    order = 'F')
                    self.Efield['fz'] = ez.reshape((self.Efield['nR'][0],
                                                    self.Efield['nPhi'][0],
                                                    self.Efield['nZ'][0],
                                                    self.Efield['nTime'][0]),
                                                    order = 'F')
                    self.Efield['ft'] = ephi.reshape((self.Efield['nR'][0],
                                                      self.Efield['nPhi'][0],
                                                      self.Efield['nZ'][0],
                                                      self.Efield['nTime'][0]),
                                                      order = 'F')

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'][0],
                                                   self.Efield['Rmax'][0],
                                                   self.Efield['nR'][0])
                    self.Efield['z'] = np.linspace(self.Efield['Zmin'][0],
                                                   self.Efield['Zmax'][0],
                                                   self.Efield['nZ'][0])
                    self.Efield['Phi'] = np.linspace(self.Efield['Phimin'][0],
                                                     self.Efield['Phimax'][0],
                                                     self.Efield['nPhi'][0])
                    self.Efield['time'] = np.linspace(self.Efield['Timemin'][0],
                                                      self.Efield['Timemax'][0],
                                                      self.Efield['nTime'][0])

                    self.Erinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z'],
                                            self.Efield['Phi'],
                                            self.Efield['time']),
                                            self.Efield['fr'],
                                            (r.flatten(), z.flatten(),
                                             phi.flatten(), time.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z'],
                                            self.Efield['Phi'],
                                            self.Efield['time']),
                                            self.Efield['fz'],
                                            (r.flatten(), z.flatten(),
                                             phi.flatten(), time.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'],
                                            self.Efield['z'],
                                            self.Efield['Phi'],
                                            self.Efield['time']),
                                            self.Efield['ft'],
                                            (r.flatten(), z.flatten(),
                                             phi.flatten(), time.flatten()))
                
                del er
                del ez
                del ephi

    def readBfromDB(self, time:float, shotnumber: int = 34570,
                     exp: str = 'AUGD', diag: str = 'EQI',
                     edition: int = 0,
                     Rmin: float = 1.03, Rmax: float = 2.65,
                     zmin: float = -1.224, zmax: float = 1.05,
                     nR: int = 128, nZ: int = 256):
        """
        Starts the class info for the magnetic field using the AUG
        database equilibrium.

        Pablo Oyola: pablo.oyola@ipp.mpg.de
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
        self.edims = 0
        self.psipol_on = False

        # Getting from the database.        
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nZ)
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
        self.Bfield['R'] = np.array(R, dtype = np.float64)
        self.Bfield['z'] = np.array(z, dtype = np.float64)
        self.Bfield['Rmin'] = np.array((Rmin), dtype = np.float64)
        self.Bfield['Rmax'] = np.array((Rmax), dtype = np.float64)
        self.Bfield['Zmin'] = np.array((zmin), dtype = np.float64)
        self.Bfield['Zmax'] = np.array((zmax), dtype = np.float64)
        self.Bfield['nR'] = np.array([nR], dtype = np.int32)
        self.Bfield['nZ'] = np.array([nZ], dtype = np.int32)
        self.Bfield['fr'] = Br.astype(dtype = np.float64)
        self.Bfield['fz'] = Bz.astype(dtype = np.float64)
        self.Bfield['ft'] = Bt.astype(dtype = np.float64)

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                 self.Bfield['fr'],
                                (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                 self.Bfield['fz'],
                                (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']),
                                 self.Bfield['ft'],
                                (r.flatten(), z.flatten()))
        
        # Retrieving as well the poloidal magnetic flux.
        self.readPsiPolfromDB(time, shotnumber = shotnumber, 
                              exp = exp, diag = diag, edition = edition, 
                              Rmin = Rmin, Rmax = Rmax, 
                              zmin = zmin, zmax = zmax,
                              nR = nR, nZ = nZ)
                        
        # Saving the input data to the class.
        self.Bfield_from_shot_flag = True
        self.shotnumber = shotnumber
        self.edition = edition
        self.timepoint = time
        self.diag = diag
        self.exp = exp
    
    def readPsiPolfromDB(self, time: float, shotnumber: int = 34570,
                          exp: str = 'AUGD', diag: str = 'EQI',
                          edition: int = 0,
                          Rmin: float = 1.03, Rmax: float = 2.65,
                          zmin: float = -1.224, zmax: float = 1.05,
                          nR: int = 128, nZ: int = 256):
        """
        Fetchs the psi_pol = rho_pol(R, z) map from the AUG database using 
        input grid.

        Jose Rueda: jrrueda@us.es
        Pablo Oyola: pablo.oyola@ipp.mpg.de
        
        @param time: Time point from which magnetic equilibrium will be read.
        @param shotnumber: Shot from which to extract the magnetic equilibria
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


        # Getting from the database.        
        R = np.linspace(Rmin, Rmax, num=nR)
        z = np.linspace(zmin, zmax, num=nZ)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        psipol = ssdat.get_psipol(shotnumber, RR.flatten(), zz.flatten(),
                                  diag=diag, time=time)

        # Reshaping into the original shape.
        psipol = np.reshape(psipol, grid_shape).T

        # Creating the interpolating function.
        self.psipol['R'] = np.array(R, dtype = np.float64)
        self.psipol['z'] = np.array(z, dtype = np.float64)
        self.psipol['nR'] = np.array([nR], dtype = np.int32)
        self.psipol['nZ'] = np.array([nZ], dtype = np.int32)
        self.psipol['f'] = psipol.astype(dtype = np.float64)
        self.psipol_interp = lambda r, z, phi, time: \
                             interpn((self.psipol['R'], self.psipol['z']), \
                                     self.psipol['f'], \
                                     (r.flatten(), z.flatten()))
        self.psipol_on = True
        return

    def getBfield(self, R: float, z: float,
                  phi: float = None, t: float = None):
        """
        Gets the magnetic field components at the given points. 

        Pablo Oyola: pablo.oyola@ipp.mpg.de

        @param R: Major radius to evaluate the magnetic field.
        @param z: Vertical position to evaluate the magnetic field.
        @param phi: Toroidal location to evaluate the magnetic field. 
        If the system is in 2D, it will be ignored.
        @param t: Time to evaluate the magnetic field. If the magnetic
        field is only stored for a single time, it will be ignored.

        @return Br: Radial component of the magnetic field.
        @return Bz: Vertical component of the magnetic field.
        @return Bphi: Toroidal component of the magnetic field.

        """
        if self.bdims != 0:
            Br   = self.Brinterp(R, z, phi, t)
            Bz   = self.Bzinterp(R, z, phi, t)
            Bphi = self.Bphiinterp(R, z, phi, t)
        else:
            raise Exception('The magnetic field has not been loaded!')

        return Br, Bz, Bphi

    def getEfield(self, R: float, z: float,
                  phi: float = None, t: float = None):
        """
        Gets the electric field components at the given points. 

        Pablo Oyola: pablo.oyola@ipp.mpg.de

        @param R: Major radius to evaluate the electric field.
        @param z: Vertical position to evaluate the electric field.
        @param phi: Toroidal location to evaluate the electric field. 
        If the system is in 2D, it will be ignored.
        @param t: Time to evaluate the electric field. If the electric
        field is only stored for a single time, it will be ignored.

        @return Er: Radial component of the magelectricnetic field.
        @return Ez: Vertical component of the electric field.
        @return Ephi: Toroidal component of the electric field.
        """
        if self.edims != 0:
            Er   = self.Erinterp(R, z, phi, t)
            Ez   = self.Ezinterp(R, z, phi, t)
            Ephi = self.Ephiinterp(R, z, phi, t)
        else:
            raise Exception('The electric field has not been loaded!')

        return Er, Ez, Ephi

    def getPsipol(self, R: float, z: float,
                  phi: float = None, t: float = None):
        """
        Gets the poloidal magnetic flux at the position.

        Pablo Oyola: pablo.oyola@ipp.mpg.de

        @param R: Major radius to evaluate the poloidal flux.
        @param z: Vertical position to evaluate the poloidal flux.
        @param phi: Toroidal location to evaluate the poloidal flux. 
        If the system is in 2D, it will be ignored.
        @param t: Time to evaluate the poloidal flux. If the poloidal
        flux is only stored for a single time, it will be ignored.

        @return psipol: Poloidal flux at the input points.
        """
        if self.psipol_on:
            psipol   = self.psipol_interp(R, z, phi, t)
        else:
            raise Exception('The poloidal flux is not loaded!')

        return psipol

    def tofile(self, fid, bflag: bool = True, eflag: bool = False):
        """
        Write the magnetic or the electric field to files following
        the i-HIBPsims scheme.

        Pablo Oyola: pablo.oyola@ipp.mpg.de

        @param fid: file identifier where the files will be written.
        @param bflag: states if the magnetic field has to be written.
        Default is True, so magnetic field will be written.
        @param eflag: states if the electric field has to be written. 
        Default to False. If this is set to True, the magnetic field 
        will not be written.
        """

        if bflag is False and eflag is False:
            raise Exception('Some flag has to be set to write to file!')
        if (eflag is True) and (self.edims == 0):
            raise Exception('Non-existent electric field in the class')
        elif (eflag is True):
             # Write header with grid size information:
            self.Efield['nR'].tofile(fid)
            self.Efield['nZ'].tofile(fid)
            self.Efield['nPhi'].tofile(fid)
            self.Efield['nTime'].tofile(fid)

            # Write grid ends:
            self.Efield['Rmin'].tofile(fid)
            self.Efield['Rmax'].tofile(fid)
            self.Efield['Zmin'].tofile(fid)
            self.Efield['Zmax'].tofile(fid)
            self.Efield['Phimin'].tofile(fid)
            self.Efield['Phimax'].tofile(fid)
            self.Efield['tmin'].tofile(fid)
            self.Efield['tmax'].tofile(fid)

            # Write fields
            self.Efield['fr'].T.tofile(fid)
            self.Efield['ft'].T.tofile(fid)
            self.Efield['fz'].T.tofile(fid)
        
        elif self.bdims != 0:
            # Write header with grid size information:
            self.Bfield['nR'].tofile(fid)
            self.Bfield['nZ'].tofile(fid)
            self.Bfield['nPhi'].tofile(fid)
            self.Bfield['nTime'].tofile(fid)

            # Write grid ends:
            self.Bfield['Rmin'].tofile(fid)
            self.Bfield['Rmax'].tofile(fid)
            self.Bfield['Zmin'].tofile(fid)
            self.Bfield['Zmax'].tofile(fid)
            self.Bfield['Phimin'].tofile(fid)
            self.Bfield['Phimax'].tofile(fid)
            self.Bfield['Timemin'].tofile(fid)
            self.Bfield['Timemax'].tofile(fid)

            # Write fields
            self.Bfield['fr'].T.tofile(fid)
            self.Bfield['ft'].T.tofile(fid)
            self.Bfield['fz'].T.tofile(fid)
        else:
            raise Exception('Not a valid combination of inputs')

    def plot(self, fieldName: str, phiSlice: int = None, timeSlice: int = None,
             ax_options: dict = {}, ax = None, cmap = None, nLevels: int = 50,
              cbar_tick_format: str = '%.2e'):
        """
        Plots the input field ('Br', 'Bz', 'Bphi', 'Er', 'Ez', 'Ephi', 
        'B', 'E' or ''Psipol') into some axis, ax, or the routine creates one 
        for the plotting.
        
        @param fieldName: Name of the field to be plotted.
        @param ax_options: options for the function axis_beauty.
        @param ax: Axis to plot the data.
        @param cmap: Colormap to use. Gamma_II is set by default.
        @param ax: Return the axis where the plot has been done.

        """
        
        # --- Initialise the plotting parameters
        ax_options['ratio'] = 'equal'
        # The ratio must be always equal
        if 'fontsize' not in ax_options:
            ax_options['fontsize'] = 16
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
        dims = 2
        if fieldName[0] == 'e':
            if self.edims == 0:
                raise Exception('Electric field is not loaded!')
            dims = self.edims
            if dims >= 3:
                phiMaxIdx = self.Bfield['nPhi']
                if dims >= 4:
                    timeMaxIdx = self.Bfield['nTime']
                    
        if fieldName[0] == 'b':
            if self.bdims == 0:
                raise Exception('Magnetic field is not loaded!')
            dims = self.bdims
            if dims >= 3:
                phiMaxIdx = self.Bfield['nPhi']
                if dims >= 4:
                    timeMaxIdx = self.Bfield['nTime']
        
        if fieldName == 'psipol' and (self.psipol_on == False):
            raise Exception('Magnetic flux is not loaded!')
        
        # Get the appropriate field
        field = {
            'br': self.Bfield['fr'],
            'bz': self.Bfield['fz'],
            'bphi': self.Bfield['ft'],
            'er': self.Efield['fr'],
            'ez': self.Efield['fz'],
            'ephi': self.Efield['ft'],
            'b': np.sqrt(self.Bfield['fr']**2 +
                         self.Bfield['fz']**2 +
                         self.Bfield['ft']**2),
            'e': np.sqrt(self.Efield['fr']**2 +
                         self.Efield['fz']**2 +
                         self.Efield['ft']**2),
            'psipol': self.psipol['f']
        }.get(fieldName)
        
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
        }.get(fieldName)
        
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
        }.get(fieldName)
        
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
        fieldPlotHndl = plt.contourf(Rplot, Zplot, field.T, 
                                     nLevels, cmap = ccmap)
        
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
            }.get(fieldName)
        
        cbar = plt.colorbar(fieldPlotHndl, format=cbar_tick_format)
        cbar.set_label(ccbarname, fontsize=ax_options['fontsize'])
        cbar.ax.tick_params(labelsize=ax_options['fontsize'] * 0.8)
        
        
        # Preparing the axis labels:
        ax_options['xlabel'] = 'Major radius R [m]'
        ax_options['ylabel'] = 'Z [m]'
        ax1 = ssplt.axis_beauty(ax, ax_options)
        
        # Plotting the 2D vessel structures.
        ssplt.plot_vessel(linewidth=1, ax=ax1)
        plt.tight_layout()
        
        
        return ax1
    
    
class ihibpProfiles:
    def __init__(self):
        """
        Initializes a dummy object. Call the class methods:
        a) readFiles: to read from the files the appropriate profiles.
        b) readDB: to fetch the data from the database.

        Pablo Oyola: pablo.oyola@ipp.mpg.de
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
                          
        