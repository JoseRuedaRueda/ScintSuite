"""
Contains a variaity of methods

Jose Rueda: jruedaru@uci.edu
"""
from venv import logger
import MDSplus
import numpy as np
import xarray as xr
import ScintSuite.errors as errors
from typing import Optional
import logging
logger = logging.getLogger('ScintSuite.D3D.FC')
# -----------------------------------------------------------------------------
# %% SIGNAL OF FAST CHANNELS.
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, shot: int,
                     channels: Optional[tuple] = None,
                     MDSserver: str = 'atlas.gat.com',
                     **kwargs):
    """
    Get the signal for the fast channels (PMT, APD)

    Jose Rueda Rueda: jruedaru@uci.edu

    :param  diag: diagnostic: 'FILD' or 'INPA'
    :param  diag_number: number of the diagnostic we want to load
    :param  channels: channel number we want, or arry with channels
    :param  shot: shot file to be opened
    :param  MDSserver: server to be used
    :param  kwargs: additional arguments, ignored, just to be compatible with AUG callings
    """
    # --- Settings check and naming
    # Check the inputs and see which channels we need to load
    suported_diag = ['fild', 'inpa']
    if diag.lower() not in suported_diag:
        raise errors.NotValidInput('No understood diagnostic')
    if channels is None:
        if diag.lower() == 'inpa' and diag_number == 1:
            if shot > 199431 and shot <=199449:
                channels = np.arange(1, 17).tolist()
            elif shot > 199449 and shot <= 200713:
                channels = np.arange(1, 13).tolist()
            elif shot > 200713:
                channels = np.arange(1, 12).tolist()
            
    else:
        channels = np.atleast_1d(channels).tolist()
    # Hardcoded parameters of signal names and locations
    signalNames = {}
    # if diag.lower() == 'inpa' and diag_number == 1:
    #     if shot > 199435 and shot <=199500:
    #         for i in range(16):
    #             signalNames[i+1] = '\FILD{:0>2}S'.format(i+1)
    #         timebaseName = '\TOP:IONS:FILD:TIMEBASE'
    #     elif shot > 199500:
    #         for i in range(8):
    #             signalNames[i+1] = '\TOP:IONS:INPA:INPA{:0>2}S'.format(i+1)
    #         for i in range(4):
    #             signalNames[i+9] = '\FILD{:0>2}S'.format(i+1)
    #         timebaseName = '\TOP:IONS:INPA:TIMEBASE'
    #     else:
    #         raise errors.NotValidInput('too old shot')
    if diag.lower() == 'inpa' and diag_number == 1:
        if shot > 199431 and shot <=199449:
            for i in range(16):
                signalNames[i+1] = 'PTDATA("FILD{:0>2}"'.format(i+1) + ', %i)'%shot
            timebaseName = 'DIM_OF(PTDATA("FILD01",%i))'%shot
        elif shot > 199449:
            for i in range(8):
                signalNames[i+1] = 'PTDATA("INPA{:0>2}"'.format(i+1) + ', %i)'%shot
            for i in range(4):
                signalNames[i+9] = 'PTDATA("FILD{:0>2}"'.format(i+1) + ', %i)'%shot
            timebaseName = 'DIM_OF(PTDATA("INPA01",%i))'%shot
        elif shot > 200713:
            for i in range(8):
                signalNames[i+1] = 'PTDATA("INPA{:0>2}"'.format(i+1) + ', %i)'%shot
            signalNames[9] = 'PTDATA("FILD01", %i)'%shot
            signalNames[10] = 'PTDATA("FILD02", %i)'%shot
            signalNames[11] = 'PTDATA("FILD04", %i)'%shot
            timebaseName = 'DIM_OF(PTDATA("INPA01",%i))'%shot
        else:
            raise errors.NotValidInput('too old shot')
    # --- Loading block
    # Open the MDSplus connection
    c = MDSplus.Connection(MDSserver)
    # c.openTree('d3d', shot)
    # Load the first channel to see the matrix we need
    logger.debug('Loading signal {}'.format(signalNames[channels[0]]))
    dummy = c.get(signalNames[channels[0]]).data()
    nt = dummy.size
    nch = len(channels)
    # Create the data array
    data = np.zeros((nch, nt))
    # Load the data
    for i, ch in enumerate(channels):
        if i == 0: # We already have this guy
            data[i, :] = dummy
        else:
            logger.debug('Loading signal {}'.format(signalNames[ch]))
            data[i, :] = c.get(signalNames[ch]).data()
    # Load the time base, move to seconds
    time = c.get(timebaseName).data() / 1000.0
    #c.closeTree('d3d', shot)
    # Transform to a dataArray
    return xr.DataArray(data, dims=('channel', 't'),
                        coords={'channel': channels, 't': time})

