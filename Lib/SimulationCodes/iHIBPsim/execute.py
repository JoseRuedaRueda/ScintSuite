"""
iHIBPsim library to execute a simulation.

Pablo Oyola - pablo.oyola@ipp.mpg.de

Contains the routines needed to run from Python the iHIBPsim codes and see
the main results (orbits, strikes, strikeline evolution...)
"""

import os
import Lib.SimulationCodes.iHIBPsim.nml as nml_lib
from Lib._Machine import machine
from Lib._Paths import Path
from copy import deepcopy

paths = Path(machine)

std_namelist = { 'elec_name': os.path.join(paths.ihibp_repo, 'tables',
                                           'Rbplus_elec.bin'),
                 'cx_name': os.path.join(paths.ihibp_repo, 'tables',
                                         'Rb_CX.bin'),
                 'prm_name': os.path.join(paths.ihibp_repo, 'tables',
                                          'Rb0_elec.bin'),
                 'triangle_file': os.path.join(paths.ihibp_repo, 'model3d',
                                               'scint_plate.3d'),
                 'scint_vertex_file': os.path.join(paths.ihibp_repo, 'plate',
                                                   'scint_v0.dat'),

               }

def prepare_run(runID: str, action: str='shot_remap', params: dict={}):
    """
    This function will create all the dependences needed to run iHIBPsim
    code.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param runID: identifier of the run.
    @param action: type of the run. To be chosen among IHIBPSIM_ACTION_NAMES.
    @param parameter: parameter dictionary. The variables not provided will
    be automatically generated, like the names of the output files.
    """
    # Checks if the folder exists:
    path = os.path.join(paths.ihibp_res, runID)
    if os.path.isdir(path):
        print('The folder associated with the ID %s exists'%runID)
        cont = ''
        while cont not in ('True', 'False'):
            cont = input('Want to replace it? (True or False)')
        if cont == 'False':
            raise Exception('Folder already existing!')

    else:
        os.mkdir(path)

    # Check if the user provided the needed files.
    if 'file_orbits' not in params:
        params['file_orbits'] = path + runID
        if action != 'shot_mapper':
            params['file_orbits'] += '.orbits'

    if 'file_strikes' not in params:
        params['file_strikes'] = os.path.join(path, runID)

    if 'file_out' not in params:
        params['file_out'] = os.path.join(path, runID+'.strikes')

    if 'depos_file' not in params:
        params['depos_file'] = os.path.join(path, runID+'.depos')

    newparams = deepcopy(std_namelist)
    newparams.update(params)

    # Generating the corresponding namelist:
    nml = nml_lib.create_namelist(action, newparams)

    # Now, we check if the namelist is properly made, including the user
    # data.
    err, msg = nml_lib.check_namelist(params=nml, codename=action)

    if err != 0:
        raise ValueError('Namelist is not properly made:'+\
                         '[ERROR:%d] %s'%(err, msg))

    # Checking that the files exists.
    flags = nml_lib.check_files(nml, action=action)
    if not flags:
        raise FileNotFoundError('Some of the initial files are not provided!')

    # Writing the namelist to file.
    nml_fn = os.path.join(path, runID + '.nml')
    nml_lib.f90nml.write(nml, nml_fn)

    return


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
    path = os.path.join(paths.ihibp_res, runID)
    nmls = [ii for ii in os.listdir(path) if ii.endswith('.cfg')]

    if len(nmls) == 0:
        raise ValueError('The runID provided does not contain any namelist!')

    # Getting the absolute path to the binary.
    bin_dir = os.path.join(paths.ihibp_bins, action+'.go')

    for ii in nmls:
        os.system(bin_dir+' '+ii)

    return