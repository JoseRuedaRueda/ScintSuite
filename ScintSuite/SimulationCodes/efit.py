"""

"""
import logging

import numpy as np
import re
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
logger = logging.getLogger('ScintSuite.EFIT')

def file_numbers(fp):
    """Generator to get numbers from a text file"""
    toklist = []
    while True:
        line = fp.readline()
        if not line: break
        # Match numbers in the line using regular expression
        pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?'
        toklist = re.findall(pattern, line)
        for tok in toklist:
            yield tok
            
class GFile:
    """Class to hold the data from a G-EQDSK file"""
    def __init__(self, f,
                 precalculateInterpolators: bool=True,
                 precalculateRhoTorInterpolator: bool=False,):
        self._data = {}
        self.shot = None
        self._readFile(f)
        if precalculateInterpolators:
            self.calculateFinterpolators()
        if precalculateRhoTorInterpolator:
            self.calculate_rhotor_to_rhopol_interpolator()
    def __getitem__(self, key):
        return self._data[key]
    def __setitem__(self, key, value):
        self._data[key] = value
        
    def _readFile(self, f):
    
        if isinstance(f, str):
            # If the input is a string, treat as file name
            with open(f) as fh: # Ensure file is closed
                return self._readFile(fh) # Call again with file object

        # Read the first line, which should contain the mesh sizes
        desc = f.readline()
        if not desc:
            raise IOError("Cannot read from input file")

        s = desc.split() # Split by whitespace
        if len(s) < 3:
            raise IOError("First line must contain at least 3 numbers")

        idum = int(s[-3])
        nw = int(s[-2])
        nh = int(s[-1])
        self['nR'] = nw
        self['nZ'] = nh
        # Use a generator to read numbers
        token = file_numbers(f)

        rdim   = float(next(token))
        self['rdim'] = rdim
        zdim   = float(next(token))
        self['zdim'] = zdim
        rcentr = float(next(token))
        self['rcentr'] = rcentr
        rleft  = float(next(token))
        self['rleft'] = rleft
        zmid   = float(next(token))
        self['zmid'] = zmid

        rmaxis = float(next(token))
        self['RMAXIS'] = rmaxis
        zmaxis = float(next(token))
        self['ZMAXIS'] = zmaxis
        simag  = float(next(token))
        self['SSIMAG'] = simag
        sibry  = float(next(token))
        self['SSIBRY'] = sibry
        bcentr = float(next(token))
        self['BCENTR'] = bcentr
        current= float(next(token))
        self['CURRENT'] = current
        simag  = float(next(token))
        xdum   = float(next(token))
        rmaxis = float(next(token))
        xdum   = float(next(token))

        zmaxis = float(next(token))
        xdum   = float(next(token))
        sibry  = float(next(token))
        xdum   = float(next(token))
        xdum   = float(next(token))

        # Read arrays
        def read_array(n, name="Unknown"):
            data = np.zeros([n])
            try:
                for i in np.arange(n):
                    data[i] = float(next(token))
            except:
                raise IOError("Failed reading array '"+name+"' of size ", n)
            return data

        # read 2d array
        def read_2d(nw, nh, name="Unknown"):
            data = np.zeros([nw, nh])
            for j in np.arange(nh):
                for i in np.arange(nw):
                    data[i,j] = float(next(token))
            return data

        fpol   = read_array(nw, "fpol")
        self['FPOL'] = fpol
        nrho = self['FPOL'].size
        pres   = read_array(nw, "pres")
        ffprim = read_array(nw, "ffprim")
        pprime = read_array(nw, "pprime")
        psirz  = read_2d(nw, nh, "psirz")
        self['PSIRZ'] = psirz
        qpsi   = read_array(nw, "qpsi")
        self['Q'] = qpsi
        self['PSI'] = np.linspace(self['SSIMAG'], self['SSIBRY'], nrho)
        self['RHOPOL'] = np.sqrt(np.abs(self['PSI']-self['SSIMAG'])/np.abs(self['SSIBRY']-self['SSIMAG']))
        # Read boundary and limiters, if present
        nbbbs  = int(next(token))
        limitr = int(next(token))

        if nbbbs > 0:
            rbbbs = np.zeros([nbbbs])
            zbbbs = np.zeros([nbbbs])
            for i in range(nbbbs):
                rbbbs[i] = float(next(token))
                zbbbs[i] = float(next(token))
        else:
            rbbbs = [0]
            zbbbs = [0]

        if limitr > 0:
            rlim = np.zeros([limitr])
            zlim = np.zeros([limitr])
            for i in range(limitr):
                rlim[i] = float(next(token))
                zlim[i] = float(next(token))
        else:
            rlim = [0]
            zlim = [0]

        # Construct R-Z mesh
        r = np.linspace(rleft, rleft + rdim, nw)
        z = np.linspace(zmid - 0.5*zdim, zmid + 0.5*zdim, nh)
        self['R'] = r
        self['Z'] = z
        self['rsep'] = rbbbs
        self['zsep'] = zbbbs
        # Create dictionary of values to return
        result = {'nw': nw, 'nh':nh,        # Number of horizontal and vertical points
                'r':r, 'z':z,                     # Location of the grid-poinst
                'rdim':rdim, 'zdim':zdim,         # Size of the domain in meters
                'rcentr':rcentr, 'bcentr':bcentr, # Reference vacuum toroidal field (m, T)
                'rleft':rleft,                  # R of left side of domain
                'zmid':zmid,                      # Z at the middle of the domain
                'rmaxis':rmaxis, 'zmaxis':zmaxis,     # Location of magnetic axis
                'ssimag':simag, # Poloidal flux at the axis (Weber / rad)
                'ssibry':sibry, # Poloidal flux at plasma boundary (Weber / rad)
                'current':current,
                'psirz':psirz.T,    # Poloidal flux in Weber/rad on grid points
                'fpol':fpol,  # Poloidal current function on uniform flux grid
                'ffprim':ffprim, # derivative of poloidal flux
                'pres':pres,  # Plasma pressure in nt/m^2 on uniform flux grid
                'pprime':pprime, # derivative of pressure
                'qpsi':qpsi,  # q values on uniform flux grid
                'nbdry':nbbbs, 'bdry':np.vstack((rbbbs,zbbbs)).T, # Plasma boundary
                'limitr':limitr, 'lim':np.vstack((rlim,zlim)).T} # Wall boundary

        return result
    
    def calculateFinterpolators(self):
        """
        Calculate the interpolators of F for the B calculation
        """
        # first deravitive of the psi in RZ direction
        dpsi_dr = np.zeros([self['nR'],self['nZ']])
        dpsi_dz = np.zeros([self['nR'],self['nZ']])
        r = self['R']
        z = self['Z']
        logger.debug(self['PSIRZ'].shape)
        psirz = self['PSIRZ']
        logger.debug(psirz.shape)
        for i in range(1, self['nR']-1):
            for j in range(1, self['nZ']-1):
                dpsi_dr[i,j] =\
                    (psirz[i+1,j]-psirz[i-1,j])/(r[i+1]-r[i-1])
                dpsi_dz[i,j] =\
                    (psirz[i,j+1]-psirz[i,j-1])/(z[j+1]-z[j-1])
        # interpolate f
        logger.debug('Getting Psi inteprolartor')
        psirz = np.array((psirz-self['SSIMAG'])/\
                         (self['SSIBRY']-self['SSIMAG']),)
        fpsi = RegularGridInterpolator((r, z), psirz, 
                                       method='cubic', fill_value=0.0,
                                       bounds_error=False)
        logger.debug('Getting dr interpolator')
        fr = RegularGridInterpolator((r, z), dpsi_dr, 
                                     method='cubic', fill_value=0.0,
                                     bounds_error=False)
        logger.debug('Getting dz interpolator')
        fz = RegularGridInterpolator((r, z), dpsi_dz, 
                                     method='cubic', fill_value=0.0,
                                     bounds_error=False)

        self['fpsi'] = fpsi
        self['fr'] = fr
        self['fz'] = fz
        self['rc'] = lambda x: self['rcentr']
        self['bc'] = lambda x: self['BCENTR']

    def calculate_rhopol_to_rhotor_interpolator(self):
        logger.debug('Getting rhotor interpolator')
        frho = RegularGridInterpolator((self['RHOPOL']), 
                                       self['RHOTOR'], 
                                method='cubic', fill_value=0.0,
                                bounds_error=False)
        self['rhopol_to_rhotor'] = frho
        return
    
    def Bfield(self, r, z):
        """
        Calculate the B field at a given time and position

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
        cpasma = self['CURRENT'] # equ in res in function bfield.prepare
        currentsign = np.sign(cpasma)
        # local magnetic field strength
        l_dpsir = fr((r, z))
        l_dpsiz = fz((r, z))
        br = currentsign*-1.0*-(1/r)*l_dpsiz
        bz = currentsign*-1.0* (1/r)*l_dpsir
        bt = np.asarray(rc(0.0)*bc(0.0)/r)
        return br, bz, bt