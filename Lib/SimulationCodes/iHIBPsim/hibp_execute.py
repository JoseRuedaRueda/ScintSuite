"""
iHIBPsim library to execute a simulation.

Pablo Oyola - pablo.oyola@ipp.mpg.de

Contains the routines needed to run from Python the iHIBPsim codes and see
the main results (orbits, strikes, strikeline evolution...)
"""

import os
import numpy as np
import Lib.SimulationCodes.iHIBPsim.nml as nml_lib
from Lib.LibMachine import machine
from Lib.LibPaths import Path

paths = Path(machine)


def prepare_run(runID: str, action: str='shot_remap', parameter: dict={}):
    """
    This function will create all the dependences needed to run iHIBPsim
    code.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param runID: identifier of the run.
    @param action: type of the run. To be chosen among IHIBPSIM_ACTION_NAMES.
    @param parameter: parameter dictionary. The variables not provided will
    be automatically generated. Particularly, the 
    """
    
    pass

def run_ihibpsim(runID: str, action: str='shot_remap'):
    """
    This routine will launch a simulation whose input data has been created
    previously and are stored under the corresponding path. The code to be
    launched will be chosen among the possible ones according to the action
    keyword.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param runID: Id of the corresponding run of iHIBPsim. The results from
    the simulation will be all of them stored in $HOME/ihibpsim/sims/runID.
    @param action: name of the code to be launched.
    """
    
    # Checking if there is any running namelist in the folder.
    path = os.path.join(paths.ihibp_res, 'runID')
    nmls = [ii for ii in os.listdir(path) if ii.endswith('.cfg')]
    
    if len(nmls) == 0:
        raise ValueError('The runID provided does not contain any namelist!')
        
    # Getting the absolute path to the binary.
    bin_dir = os.path.join(paths.ihibp_bins, action+'.go')
    
    for ii in nmls:
        os.system(bin_dir+' '+ii)
        
    return