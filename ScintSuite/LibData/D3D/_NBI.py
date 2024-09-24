"""
Module to get the NBI data
"""
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from ScintSuite.LibData.D3D import _gadata
logger = logging.getLogger('ScintSuite.data.NBI')
# -----------------------------------------------------------------------------
# %% NBI Object
# -----------------------------------------------------------------------------
class NBI:
    """
    NBI Object
    """
    def __init__(self, shot):
        """
        Inintialize the NBI object
        
        :param shot: Shot number
        """
        self.shot = shot
        try:
            self.getPower()
        except:
            self.power = None
        
        try:
            self.getVoltaje()
        except:
            self.voltaje = None
        
    def getVoltaje(self):
        """
        Get the NBI power
        """
        sig = ['pcnbv30lt',
               'pcnbv30rt',
               'pcnbv15lt',
               'pcnbv15rt',
               'pcnbv21lt',
               'pcnbv21rt',
               'pcnbv33lt',
               'pcnbv33rt' ]
        logger.debug('Getting NBI power')
        logger.debug('Reading signal: '+sig[0])
        tmp = _gadata.gadata(sig[0],self.shot)
        nb30r = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[1],self.shot)
        nb30l = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[2],self.shot)
        nb15r = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[3],self.shot)
        nb15l = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[4],self.shot)
        nb21r = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[5],self.shot)
        nb21l = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[6],self.shot)
        nb33r = tmp.zdata
        logger.debug('Reading signal: '+sig[0])
        
        tmp = _gadata.gadata(sig[7],self.shot)
        nb33l = tmp.zdata
        nbtime = tmp.xdata/1e3
        # Move to a dataArray
        x = np.array(['30L', '30R', '150L', '150R', '210L', '210R', '330L', '330R'])
        # Concatenate all the powers in a single array
        y = np.array([nb30l, nb30r, nb15l, nb15r, nb21l, nb21r, nb33l, nb33r])
        print(y.shape)
        # Create the dataArray
        self.voltaje = xr.DataArray(y/1000.0, dims=['NBI', 't'], coords={'NBI':x, 't':nbtime}, )

    def getPower(self):
        self.power = None
    
    def plot330(self, ax=None):
        """
        Plot the NBI power at 330kV
        
        :param ax: Axis to plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        if self.voltaje is not None:
            a = self.voltaje.sel(NBI='330L')
            a.plot(ax=ax, label='330L', ls='--', color='r')
            b = self.voltaje.sel(NBI='330R')
            b.plot(ax=ax, label='330R', ls='--', color='b')
            (a+b).plot(ax=ax, label='330L+330R', color='k')
            
            ax.legend()
        else:
            print('No power data available')
            
