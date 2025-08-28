"""
Read and plot ALCON output files
"""
import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ScintSuite._Utilities import flatten
from ScintSuite._MHD import MHDmode
from scipy.cluster.vq import kmeans2
from sklearn.cluster import SpectralClustering, DBSCAN, OPTICS

logger = logging.getLogger('ScintSuite.ALCON')

class ALCON:
    def __init__(self, folder, settingsExt='.txt'):
        self.folder = folder
        self._read_alcon_settings(settingsExt)
        self._read_alcon_files()
    
    def _read_alcon_settings(self, fileExt = '.txt'):
        """
        Find and read the txt file with the settings
        
        :param fileExt: The extension of the settings file, default is '.txt'
        """
        self.settings = None
        for file in os.listdir(self.folder):
            if file.endswith(fileExt):
                # Try to read the file
                try:
                    with open(os.path.join(self.folder, file), 'r') as f:
                        self.settings = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    continue
                # If successful, set the filename and break
                self.filenameSettings = os.path.join(self.folder, file)
                break
        if self.settings is None:
            raise FileNotFoundError(f"No settings file found in {self.folder} with extension {fileExt}")
        if self.settings['PLOT']['radial_coordinate'] != 'rho':
            logger.error('the radial coordinate is not rho, but rho will be assumed!')
    
    def _read_alcon_files(self):
        """
        Read the ALCON output files from the specified folder.
        
        """
        self._data = xr.Dataset()
        omegascale = np.abs(self.settings['PHYSICS']['omega_A'])
        for file in os.listdir(self.folder):
            if file.endswith('.dat'):
                name, ext = os.path.splitext(file)
                logger.debug(f"Reading ALCON file: {file}")
                file_path = os.path.join(self.folder, file)
        
                col1, col2 = np.loadtxt(file_path, unpack=True)
                col2 *= omegascale  # Scale the second column by omega_A
                self._data[name] =\
                    xr.DataArray(col2, coords={'rho_%s'%name: col1},        
                                 dims='rho_%s'%name)
                self._data[name].attrs['units'] = 'rad/s'
                
    def _guess_n(self, name):
        """
        Guess the n number from the string.
        """
        # The first character is n, and there are 3 digits
        number = name[1:4]
        if number[0] == '_':
            factor = -1
        else:
            factor = 1
        return factor * int(number.replace('_', ''))
    
    def _guess_m(self, name):
        """
        Guess the m number from the string.
        """
        # The first character is m, and there are 3 digits
        number = name[5:8]
        if number[0] == '_':
            factor = -1
        else:
            factor = 1
        return factor * int(number.replace('_', ''))
             
    def plot(self, ax=None, n: int = None,
             flim=(0, 200), unit='kHz', color='k'):
        """
        Plot the ALCON data on the given axes.
        
        :param ax: The axes to plot on, if None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created = True
        else:
            fig = ax.figure
            created = False
        
        factor = {'rad/s': 1.0, 'Hz': 1.0/2.0/np.pi, 'kHz': 1.0/2.0/np.pi/1000.0, 'krad/s': 1/1000.0}[unit]
        
        for var in self._data.keys():
            if not var.startswith('rho_'):
                
                if n is not None:
                    if self._guess_n(var) != n:
                        continue
                (self._data[var]*factor).plot(ax=ax, marker='o', linestyle='None', color=color)
        
        if created:
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('f [%s]'%unit)
            ax.set_ylim(flim)
        return ax
    
    def findGaps(self, shot: int, time: float, n=None, omegamax = 1.25e6, 
                 MHDparams={}):
        """
        Find the gaps in the Alfven continuum for a given n.
        
        :param n: The toroidal mode number to consider, if None, all modes are considered.
        
        """
        # collect all the data in an array
        data = []
        rho = []
        for var in self._data.keys():
            if not var.startswith('rho_'):
                if n is not None:
                    if self._guess_n(var) != n:
                        continue
                data.append(self._data[var].values)
                rho.append(self._data[var].coords['rho_%s'%var].values)
        data = np.array(flatten(data))
        rho = np.array(flatten(rho))
        flags = np.abs(data) < omegamax
        data = data[flags]
        rho = rho[flags]
        
        mhd = MHDmode(shot=shot, **MHDparams)
        fgap = mhd.freq.TAE.sel(t=time, method='nearest').interp(rho=rho, method='linear').values*2*np.pi
        flags0 = np.where(fgap > data)[0]
        flags1 = np.where((fgap < data) * (data < 2.0*fgap))[0]
        flags2 = np.where(data > 2.0*fgap)[0]
        rho0 = rho[flags0]
        rho1 = rho[flags1]
        rho2 = rho[flags2]
        data0 = data[flags0]
        data1 = data[flags1]
        data2 = data[flags2]
        
        sorted_indices0 = np.argsort(rho0)
        sorted_indices1 = np.argsort(rho1)
        sorted_indices2 = np.argsort(rho2)
        
        rho0 = rho0[sorted_indices0]
        data0 = data0[sorted_indices0]
        rho1 = rho1[sorted_indices1]
        data1 = data1[sorted_indices1]
        rho2 = rho2[sorted_indices2]
        data2 = data2[sorted_indices2]
        # Create a dictionary with the gaps
        gapsLimits = {}
        if len(rho0) > 0:
            gapsLimits[0] = xr.DataArray(data0, coords={'rho': rho0}, dims='rho')
        if len(rho1) > 0:
            gapsLimits[1] = xr.DataArray(data1, coords={'rho': rho1}, dims='rho')
        if len(rho2) > 0:
            gapsLimits[2] = xr.DataArray(data2, coords={'rho': rho2}, dims='rho')
        return gapsLimits
    
    
     
        