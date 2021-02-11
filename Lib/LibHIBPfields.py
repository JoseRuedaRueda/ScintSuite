"""Contains the routines to read and write the
fields and profiles from i-HIBPsim."""

import numpy as np
from version_suite import version
from LibMachine import machine
import LibPlotting as ssplt
import LibParameters as sspar
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as intp2
from scipy.interpolate import interpn
if machine == 'AUG':
    import LibDataAUG as ssdat

class ihibpEMfields:
    def __init__(self, Bfield: dict = None, Efield: dict = None):
        """
        Initializes a dummy object. Call the class methods:
        a) readFiles: to read from the files the EM fields.
        b) readBfromAUG: to fetch the fields from the AUG database.

        Pablo Oyola: pablo.oyola@ipp.mpg.de
        """
        self.bdims = 0
        self.edims = 0
        self.psipol_on = False

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
        --> Bz[nR, nPhi, nZ, nTime]: float64

        Pablo Oyola: pablo.oyola@ipp.mpg.de

        @param Bfile: Full path to the magnetic field.
        @param Efile: Full path to the electric field.
        """

        self.bdims = 0
        self.edims = 0
        self.psipol_on = False

        if Bfile is not None:
            with open(file = Bfile, 'rb') as fid:
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

                br = np.fromfile(fid, 'float64', size2read)
                bphi = np.fromfile(fid, 'float64', size2read)
                bz = np.fromfile(fid, 'float64', size2read)

                if self.Bfield['nPhi'] == 1 and self.Bfield['nTime'] == 1:
                    self.bdims = 2
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'], \
                                                    self.Bfield['nZ'])))
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'], \
                                                    self.Bfield['nZ']))
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'], \
                                                        self.Bfield['nZ']))

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'], \
                                                   self.Bfield['Rmax'], \
                                                   self.Bfield['nR'])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'], \
                                                   self.Bfield['Zmax'], \
                                                   self.Bfield['nZ'])

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
                else if self.Bfield['nTime'] == 1:
                    self.bdims = 3
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'], \
                                                    self.Bfield['nPhi'], \
                                                    self.Bfield['nZ']))
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'], \
                                                    self.Bfield['nPhi'], \
                                                    self.Bfield['nZ']))
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'], \
                                                        self.Bfield['nPhi'], \
                                                        self.Bfield['nZ']))

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'], \
                                                   self.Bfield['Rmax'], \
                                                   self.Bfield['nR'])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'], \
                                                   self.Bfield['Zmax'], \
                                                   self.Bfield['nZ'])
                    self.Bfield['phi'] = np.linspace(self.Bfield['Phimin'], \
                                                     self.Bfield['Phimax'], \
                                                     self.Bfield['nPhi'])

                    self.Brinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z'], \
                                            self.Bfield['Phi']), \
                                            self.Bfield['fr'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z'], \
                                            self.Bfield['Phi']), \
                                            self.Bfield['fz'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z'], \
                                            self.Bfield['Phi']), \
                                            self.Bfield['ft'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten()))
                else:
                    self.bdims = 4
                    self.Bfield['fr'] = br.reshape((self.Bfield['nR'], \
                                                    self.Bfield['nPhi'], \
                                                    self.Bfield['nZ'], \
                                                    self.Bfield['nTime']))
                    self.Bfield['fz'] = bz.reshape((self.Bfield['nR'], \
                                                    self.Bfield['nPhi'],, \
                                                    self.Bfield['nZ'] \
                                                    self.Bfield['nTime']))
                    self.Bfield['ft'] = bphi.reshape((self.Bfield['nR'], \
                                                        self.Bfield['nPhi'],, \
                                                        self.Bfield['nZ'] \
                                                        self.Bfield['nTime']))

                    self.Bfield['R'] = np.linspace(self.Bfield['Rmin'], \
                                                   self.Bfield['Rmax'], \
                                                   self.Bfield['nR'])
                    self.Bfield['z'] = np.linspace(self.Bfield['Zmin'], \
                                                   self.Bfield['Zmax'], \
                                                   self.Bfield['nZ'])
                    self.Bfield['phi'] = np.linspace(self.Bfield['Phimin'], \
                                                     self.Bfield['Phimax'], \
                                                     self.Bfield['nPhi'])
                    self.Bfield['time'] = np.linspace(self.Bfield['Timemin'], \
                                                      self.Bfield['Timemax'], \
                                                      self.Bfield['nTime'])

                    self.Brinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z'], \
                                            self.Bfield['Phi'], \
                                            self.Bfield['time']), \
                                            self.Bfield['fr'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten(), time.flatten()))
                    self.Bzinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z'], \
                                            self.Bfield['Phi']), \
                                            self.Bfield['time']), \
                                            self.Bfield['fz'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten(), time.flatten()))
                    self.Bphiinterp = lambda r, z, phi, time: \
                                   interpn((self.Bfield['R'], \
                                            self.Bfield['z'], \
                                            self.Bfield['Phi']), \
                                            self.Bfield['time']), \
                                            self.Bfield['ft'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten(), time.flatten()))
                
                del br
                del bz
                del bphi
        if Efile is not None:
            with open(file = Efile, 'rb') as fid:
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

                er = np.fromfile(fid, 'float64', size2read)
                ephi = np.fromfile(fid, 'float64', size2read)
                ez = np.fromfile(fid, 'float64', size2read)

                if self.Efield['nPhi'] == 1 and self.Efield['nTime'] == 1:
                    self.edims = 2
                    self.Efield['fr'] = br.reshape((self.Efield['nR'], \
                                                    self.Efield['nZ'])))
                    self.Efield['fz'] = bz.reshape((self.Efield['nR'], \
                                                    self.Efield['nZ']))
                    self.Efield['ft'] = bphi.reshape((self.Efield['nR'], \
                                                        self.Efield['nZ']))

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'], \
                                                   self.Efield['Rmax'], \
                                                   self.Efield['nR'])
                    self.Efield['z'] = np.linspace(self.Efield['Zmin'], \
                                                   self.Efield['Zmax'], \
                                                   self.Efield['nZ'])

                    self.Erinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z']), \
                                            self.Efield['fr'], \
                                            (r.flatten(), z.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z']), \
                                            self.Efield['fz'], \
                                            (r.flatten(), z.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z']), \
                                            self.Efield['ft'], \
                                            (r.flatten(), z.flatten()))
                else if self.Efield['nTime'] == 1:
                    self.edims = 3
                    self.Efield['fr'] = br.reshape((self.Efield['nR'], \
                                                    self.Efield['nPhi'], \
                                                    self.Efield['nZ']))
                    self.Efield['fz'] = bz.reshape((self.Efield['nR'], \
                                                    self.Efield['nPhi'], \
                                                    self.Efield['nZ']))
                    self.Efield['ft'] = bphi.reshape((self.Efield['nR'], \
                                                        self.Efield['nPhi'], \
                                                        self.Efield['nZ']))

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'], \
                                                   self.Efield['Rmax'], \
                                                   self.Efield['nR'])
                    self.Efield['z'] = np.linspace(self.Efield['Zmin'], \
                                                   self.Efield['Zmax'], \
                                                   self.Efield['nZ'])
                    self.Efield['phi'] = np.linspace(self.Efield['Phimin'], \
                                                     self.Efield['Phimax'], \
                                                     self.Efield['nPhi'])

                    self.Erinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z'], \
                                            self.Efield['Phi']), \
                                            self.Efield['fr'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z'], \
                                            self.Efield['Phi']), \
                                            self.Efield['fz'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z'], \
                                            self.Efield['Phi']), \
                                            self.Efield['ft'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten()))
                else:
                    self.edims = 4
                    self.Efield['fr'] = br.reshape((self.Efield['nR'], \
                                                    self.Efield['nPhi'], \
                                                    self.Efield['nZ'], \
                                                    self.Efield['nTime']))
                    self.Efield['fz'] = bz.reshape((self.Efield['nR'], \
                                                    self.Efield['nPhi'],, \
                                                    self.Efield['nZ'] \
                                                    self.Efield['nTime']))
                    self.Efield['ft'] = bphi.reshape((self.Efield['nR'], \
                                                        self.Efield['nPhi'],, \
                                                        self.Efield['nZ'] \
                                                        self.Efield['nTime']))

                    self.Efield['R'] = np.linspace(self.Efield['Rmin'], \
                                                   self.Efield['Rmax'], \
                                                   self.Efield['nR'])
                    self.Efield['z'] = np.linspace(self.Efield['Zmin'], \
                                                   self.Efield['Zmax'], \
                                                   self.Efield['nZ'])
                    self.Efield['phi'] = np.linspace(self.Efield['Phimin'], \
                                                     self.Efield['Phimax'], \
                                                     self.Efield['nPhi'])
                    self.Efield['time'] = np.linspace(self.Efield['Timemin'], \
                                                      self.Efield['Timemax'], \
                                                      self.Efield['nTime'])

                    self.Erinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z'], \
                                            self.Efield['Phi'], \
                                            self.Efield['time']), \
                                            self.Efield['fr'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten(), time.flatten()))
                    self.Ezinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z'], \
                                            self.Efield['Phi']), \
                                            self.Efield['time']), \
                                            self.Efield['fz'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten(), time.flatten()))
                    self.Ephiinterp = lambda r, z, phi, time: \
                                   interpn((self.Efield['R'], \
                                            self.Efield['z'], \
                                            self.Efield['Phi']), \
                                            self.Efield['time']), \
                                            self.Efield['ft'], \
                                            (r.flatten(), z.flatten(), \
                                             phi.flatten(), time.flatten()))
                
                del er
                del ez
                del ephi

    def readBfromAUG(self, time:float, shotnumber: int = 34570,  \
                  exp: str = 'AUGD', diag: str = 'EQI', \
                  edition: int = 0, \ 
                  Rmin: float = 1.03, Rmax: float = 2.65, \
                  zmin: float = -1.224, zmax: float = 1.05, \
                  nr: int = 128, nZ: int = 256):
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
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        br, bz, bt, bp = ssdat.get_mag_field(shot, RR.flatten(), zz.flatten(),
                                            diag=diag, time=time)
        del RR
        del zz
        del bp
        Br = np.reshape(br, grid_shape).T
        Bz = np.reshape(bz, grid_shape).T
        Bt = np.reshape(bt, grid_shape).T
        del br
        del bt
        del bz

        # Storing the data in the class.
        self.bdims = 2
        self.Bfield['R'] = R
        self.Bfield['Z'] = z
        self.Bfield['nR'] = nR
        self.Bfield['nZ'] = nZ
        self.Bfield['fr'] = Br
        self.Bfield['fz'] = Bz
        self.Bfield['ft'] = Bt

        del Br
        del Bz
        del Bt

        # Creating the interpolating functions.
        self.Brinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']), \
                                self.Bfield['fr'], \
                                (r.flatten(), z.flatten()))

        self.Bzinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']), \
                                self.Bfield['fz'], \
                                (r.flatten(), z.flatten()))

        self.Bphiinterp = lambda r, z, phi, time: \
                        interpn((self.Bfield['R'], self.Bfield['z']), \
                                self.Bfield['ft'], \
                                (r.flatten(), z.flatten()))
    
    def readRhoPolfromAUG(self, shotnumber: int = 34570, exp: str = 'AUGD', \
                  diag: str = 'EQI', edition: int = 0, \
                  Rmin: float = 1.03, Rmax: float = 2.65, \
                  zmin: float = -1.224, zmax: float = 1.05, \
                  nR: int = 128, nz: int = 256):
        """
        Fetchs the rho_pol = rho_pol(R, z) map from the AUG database using 
        input grid.

        Jose Rueda: jrrueda@us.es
        Pablo Oyola: pablo.oyola@ipp.mpg.de

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
        z = np.linspace(zmin, zmax, num=nz)
        RR, zz = np.meshgrid(R, z)
        grid_shape = RR.shape
        psipol = ssdat.get_psipol(shot, RR.flatten(), zz.flatten(),
                                            diag=diag, time=time)

        # Reshaping into the original shape.
        psipol = np.reshape(psipol, grid_shape).T

        # Creating the interpolating function.
        self.psipol['R'] = R
        self.psipol['z'] = z
        self.psipol['nR'] = nR
        self.psipol['nz'] = nz
        self.psipol['f']  = psipol
        self.psipol_interp = lambda r, z, phi, time: \
                             interpn((self.psipol['R'], self.psipol['z']), \
                                     self.psipol['f'], \
                                     (r.flatten(), z.flatten()))
        return

    def getBfield(self, R: float, z: float, \
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
        if self.bdims is not 0:
            Br   = self.Brinterp(R, z, phi, t)
            Bz   = self.Bzinterp(R, z, phi, t)
            Bphi = self.Bphiinterp(R, z, phi, t)
        else:
            raise Exception('The magnetic field has not been loaded!')

        return Br, Bz, Bphi

    def getEfield(self, R: float, z: float, \
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
        if self.edims is not 0:
            Er   = self.Erinterp(R, z, phi, t)
            Ez   = self.Ezinterp(R, z, phi, t)
            Ephi = self.Ephiinterp(R, z, phi, t)
        else:
            raise Exception('The electric field has not been loaded!')

        return Er, Ez, Ephi

    def getPsipol(self, R: float, z: float, \
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
        else if (eflag is True):
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
            self.Efield['tmin'].tofile(fid)
            self.Efield['tmax'].tofile(fid)

            # Write fields
            self.Efield['fr'].T.tofile(fid)
            self.Efield['ft'].T.tofile(fid)
            self.Efield['fz'].T.tofile(fid)
        
        else self.bdims is not 0:
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
            self.Bfield['fr'].T.tofile(fid)
            self.Bfield['ft'].T.tofile(fid)
            self.Bfield['fz'].T.tofile(fid)
        else:
            raise Exception('Not a valid combination of inputs')

    