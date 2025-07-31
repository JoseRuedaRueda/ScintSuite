"""
Contains the routines to easily execute the SINPA code

Jose Rueda: jrrueda@us.es

Introduced in version 6.0.0
"""
import os
import f90nml
import logging
import numpy as np
# import math Lib/SimulationCodes/Common
import ScintSuite.SimulationCodes.SINPA._reading as reading
from ..Common import fields as simcomFields
import ScintSuite.SimulationCodes.Common.geometry as simcomGeo
from ScintSuite._Machine import machine
from ScintSuite._Paths import Path
import ScintSuite.errors as errors
from ScintSuite._SideFunctions import update_case_insensitive
paths = Path(machine)
logger = logging.getLogger('ScintSuite.SINPA')

def guess_strike_map_name(phi: float, theta: float, geomID: str = 'AUG02',
                          decimals: int = 1):
    """
    Give the name of the strike-map file

    Jose Rueda Rueda: jrrueda@us.es

    Names are supposed to follow the ScintSuite criteria:
    name = geomID +\
        "_map_{0:09.5f}_{1:010.5f}_strike_map.dat".format(p, t)

    :param  phi: phi angle as defined in FILDSIM
    :param  theta: theta angle as defined in FILDSIM
    :param  geomID: ID identifying the geometry
    :param  decimals: number of decimal numbers to round the angles

    :return name: the name of the strike map file
    """
    p = round(phi, ndigits=decimals)
    t = round(theta, ndigits=decimals)
    if p < 0:
        if t < 0:
            name = geomID +\
                "_map_{0:010.5f}_{1:010.5f}.map".format(p, t)
        else:
            name = geomID +\
                "_map_{0:010.5f}_{1:09.5f}.map".format(p, t)
    else:
        if t < 0:
            name = geomID +\
                "_map_{0:09.5f}_{1:010.5f}.map".format(p, t)
        else:
            name = geomID +\
                "_map_{0:09.5f}_{1:09.5f}.map".format(p, t)
    return name


def find_strike_map_FILD(phi: float, theta: float, strike_path: str,
                         geomID: str = 'AUG02',
                         SINPA_options={}, clean: bool = True,
                         decimals: int = 1):
    """
    Find the proper strike map. If not there, create it

    Jose Rueda Rueda: jrrueda@us.es

    :param     phi: phi angle as defined in FILDSIM
    :param     theta: beta angle as defined in FILDSIM
    :param     strike_path: path of the folder with the strike maps
    :param     geomID: string identifying the geometry. Defaults to 'AUG02'.
    :param     SINPA_options: FILDSIM namelist options
    :param     clean: True: eliminate the strike_points.dat when calling FILDSIM
    :param     decimals: Number of decimals for theta and phi angles

    :return   name:  name of the strikemap to load

    @raises   Exception: If FILDSIM is call but the file is not created.
    """
    # --- Find the name of the strike map
    name = guess_strike_map_name(phi, theta, geomID=geomID,
                                 decimals=decimals)
    # --- See if the strike map exist
    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # --- If do not exist, create it:
    # Guess the runID of the map
    runID = name[:-4]

    # load reference namelist
    nml = f90nml.read(os.path.join(strike_path, 'parameters.cfg'))
    # If a SINPA namelist was given, overwrite reference parameters with the
    # desired by the user, else just set at least the proper directories
    # This could be a problem because it can cause the new strike map to be
    # calculated with a geometry different from the one used for the database
    # but as the geometry directory will be different between different users
    # (different absolute paths) this is the simplest way to do it
    # If the user keeps the folder structure for the remaps, it should not be
    # an issue (famous last words)
    geom_path = os.path.join(paths.SINPA, 'Geometry', geomID)
    run_path = os.path.join(paths.SINPA, 'runs', runID)

    if SINPA_options is not None:
        # Set the geometry directory
        if 'config' in SINPA_options:
            if 'geomfolder' not in SINPA_options['config']:
                SINPA_options['config']['geomfolder'] = geom_path
        else:
            nml['config']['geomfolder'] = geom_path
        # Set a run directory.
        if 'config' in SINPA_options:
            if 'runfolder' not in SINPA_options['config']:
                SINPA_options['config']['runfolder'] = run_path
        else:
            nml['config']['runfolder'] = run_path
        # set the rest of user defined options
        for block in SINPA_options.keys():
            nml[block].update(SINPA_options[block])
    else:
        nml['config']['runfolder'] = run_path
        nml['config']['geomfolder'] = geom_path
    # set namelist name
    nml['config']['runid'] = runID
    # Write the namelist
    conf_file = write_namelist(nml)
    # Load the geometry to get the unitary vectors
    Geometry = simcomGeo.Geometry(nml['config']['geomfolder'], code='SINPA')
    u1 = np.array(Geometry.ExtraGeometryParams['u1'])
    u2 = np.array(Geometry.ExtraGeometryParams['u2'])
    u3 = np.array(Geometry.ExtraGeometryParams['u3'])
    # Prepare the field
    field = simcomFields.fields()
    field.createHomogeneousFieldThetaPhi(theta, phi, field='B',
                                         u1=u1, u2=u2, u3=u3,
                                         verbose=False, diagnostic='FILD')
    inputsDir = os.path.join(nml['config']['runfolder'], 'inputs')
    resultsDir = os.path.join(nml['config']['runfolder'], 'results')

    fieldFileName = os.path.join(inputsDir, 'field.bin')
    fid = open(fieldFileName, 'wb')
    field.tofile(fid)
    fid.close()
    # run the SINPA simulation
    bin_file = os.path.join(paths.SINPA, 'bin', 'SINPA.go')
    print('Executing SINPA')
    os.system(bin_file + ' ' + conf_file)

    # -- Move and clean
    smapName = os.path.join(resultsDir, name)
    command = 'cp ' + smapName + ' ' + strike_path
    os.system(command)

    if clean:
        command = 'rm -r ' + nml['config']['runfolder']
        os.system(command)

    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # If we reach this point, something went wrong
    a = 'SINPA simulation has been done but the strike map cannot be found'
    raise Exception(a)


def find_strike_map_INPA(phi: float, theta: float, strike_path: str,
                         s1: np.ndarray, s2: np.ndarray, s3: np.ndarray,
                         geomID: str = 'iAUG01',
                         SINPA_options={}, clean: bool = True,
                         decimals: int = 1):
    """
    Find the proper strike map. If not there, create it

    Jose Rueda Rueda: jrrueda@us.es

    :param     phi: phi angle as defined in FILDSIM
    :param     theta: beta angle as defined in FILDSIM
    :param     strike_path: path of the folder with the strike maps
    :param     geomID: string identifying the geometry. Defaults to 'AUG02'.
    :param     SINPA_options: FILDSIM namelist options
    :param     clean: True: eliminate the strike_points.dat when calling FILDSIM
    :param     decimals: Number of decimals for theta and phi angles

    :return   name:  name of the strikemap to load

    @raises   Exception: If FILDSIM is call but the file is not created.
    """
    # --- Find the name of the strike map
    name = guess_strike_map_name(phi, theta, geomID=geomID,
                                 decimals=decimals)
    # --- See if the strike map exist
    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # --- If do not exist, create it:
    # Guess the runID of the map
    runID = name[:-4]

    # load reference namelist
    nml = f90nml.read(os.path.join(strike_path, 'parameters.cfg'))
    # If a SINPA namelist was given, overwrite reference parameters with the
    # desired by the user, else just set at least the proper directories
    # This could be a problem because it can cause the new strike map to be
    # calculated with a geometry different from the one used for the database
    # but as the geometry directory will be different between different users
    # (different absolute paths) this is the simplest way to do it
    # If the user keeps the folder structure for the remaps, it should not be
    # an issue (famous last words)
    geom_path = os.path.join(paths.SINPA, 'Geometry', geomID)
    run_path = os.path.join(paths.SINPA, 'runs', runID)

    if SINPA_options is not None:
        # Set the geometry directory
        if 'config' in SINPA_options:
            if 'geomfolder' not in SINPA_options['config']:
                SINPA_options['config']['geomfolder'] = geom_path
        else:
            nml['config']['geomfolder'] = geom_path
        # Set a run directory.
        if 'config' in SINPA_options:
            if 'runfolder' not in SINPA_options['config']:
                SINPA_options['config']['runfolder'] = run_path
        else:
            nml['config']['runfolder'] = run_path
        # set the rest of user defined options
        for block in SINPA_options.keys():
            nml[block].update(SINPA_options[block])
    else:
        nml['config']['runfolder'] = run_path
        nml['config']['geomfolder'] = geom_path
    # set namelist name
    nml['config']['runid'] = runID
    # Write the namelist
    conf_file = write_namelist(nml)

    # Prepare the field
    field = simcomFields.fields()
    field.createHomogeneousFieldThetaPhi(theta, phi, field='B',
                                         u1=s1, u2=s2, u3=s3,
                                         verbose=False, diagnostic='INPA')
    inputsDir = os.path.join(nml['config']['runfolder'], 'inputs')
    resultsDir = os.path.join(nml['config']['runfolder'], 'results')

    fieldFileName = os.path.join(inputsDir, 'field.bin')
    fid = open(fieldFileName, 'wb')
    field.tofile(fid)
    fid.close()
    # run the SINPA simulation
    bin_file = os.path.join(paths.SINPA, 'bin', 'SINPA.go')
    print('Executing SINPA')
    os.system(bin_file + ' ' + conf_file)

    # -- Move and clean
    smapName = os.path.join(resultsDir, name)
    command = 'cp ' + smapName + ' ' + strike_path
    os.system(command)

    if clean:
        command = 'rm -r ' + nml['config']['runfolder']
        os.system(command)

    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # If we reach this point, something went wrong
    a = 'SINPA simulation has been done but the strike map cannot be found'
    raise Exception(a)


def write_namelist(nml, p=None, overwrite=True, grid_correction=False):
    """
    Write fortran namelist

    jose rueda: jrrueda@us.es
    Alex Reyner: alereyvinn@alum.us.es

    Also create the file structure needed to launch the simulation

    :param  nml: namelist containing the desired fields.
    :param  p: full path towards the run directory for SINPA. In principle it
        will take it from the path of the namelist. Please do not use this input
        except you really know what are you doing and want to change something
    :param  overwrite: flag to overwrite the namelist (if exist)
    :param  grid_correction: flag to automatically correct errors of the input
        regarding the grid of the strike map

    :return file: The path to the written file
    """
    # --- Check the namelist
    keys_lower_input = [key.lower() for key in nml['inputParams'].keys()]
    keys_lower_config = [key.lower() for key in nml['config'].keys()]
    keys_input = [key for key in nml['inputParams'].keys()]
    keys_config = [key for key in nml['config'].keys()]

    # --- Automatic filling of relevant parameters of the mesh
    if grid_correction == True:
        logger.warning('11: Autocorrecting the number of gyroradii and xi')
        dummy=nml
        for ik, k in enumerate(keys_lower_input):
            if k == 'rl':
                dummy['inputParams'][keys_input[ik]] =\
                      np.unique(nml['inputParams'][keys_input[ik]])
                dummy['config']['ngyroradius'] =\
                      len(nml['inputParams'][keys_input[ik]])
            elif k == 'xi':
                dummy['inputParams'][keys_input[ik]] =\
                      np.unique(nml['inputParams'][keys_input[ik]])
                dummy['config']['nxi'] =\
                      len(nml['inputParams'][keys_input[ik]])
        update_case_insensitive(nml,dummy) 

    # Check gyr and xi, adn run ID
    for ik, k in enumerate(keys_lower_config):
        if k == 'ngyroradius':
            for ik2, k2 in enumerate(keys_lower_input):
                if k2 == 'rl':
                    noMecabeFlag = nml['config'][keys_config[ik]] != \
                        len(nml['inputParams'][keys_input[ik2]])
                    if noMecabeFlag:
                        raise errors.WrongNamelist('Revise n of gyroradius')
        elif k == 'nxi':
            for ik2, k2 in enumerate(keys_lower_input):
                if k2 == 'xi':
                    noMecabeFlag = nml['config'][keys_config[ik]] != \
                        len(nml['inputParams'][keys_input[ik2]])
                    if noMecabeFlag:
                        logger.error('Size of XI is %i while nxi is %i'%(
                            len(nml['inputParams'][keys_input[ik2]]),
                            nml['config'][keys_config[ik]]
                        ))
                        raise errors.WrongNamelist('Revise n of xi')
        elif k == 'runid':
            if len(nml['config'][keys_config[ik]]) > 50:
                raise errors.WrongNamelist('runID is too long, reduce it!')
    # Initialise the path
    if p is None:
        p = nml['config']['runfolder']
    # Create if needed the directories
    os.makedirs(p, exist_ok=True)  # root directory
    inputsDir = os.path.join(p, 'inputs')
    os.makedirs(inputsDir, exist_ok=True)  # root directory
    os.makedirs(os.path.join(p, 'results'), exist_ok=True)
    # Get the full name of the namelist to be saved
    file = os.path.join(inputsDir, nml['config']['runid'] + '.cfg')
    f90nml.write(nml, file, force=overwrite)
    return file


def check_files(runID: str):
    """
    Check that the necesary files to execute SINPA exist

    Jose Rueda Rueda: jrrueda@us.es

    Note that this function will not check the files itself, just it will look
    that the files exist

    @ToDo: Clean all binary files

    :param  runID: RunID of the simulation

    :return go: Flag to say if we are ready to launch SINPA
    """
    path = os.path.join(paths.SINPA, 'runs', runID, 'inputs')
    go = True
    # --- Check namelist
    nml_ok = os.path.isfile(os.path.join(path, runID + '.cfg'))
    if not nml_ok:
        print('Namelist not found!')
        go = False
    # --- Check the magnetic field
    field_ok = os.path.isfile(os.path.join(path, 'field.bin'))
    if not field_ok:
        print('Field file not found!')
        go = False
    return go


def executeRun(runID: str = None, queue: bool = False, cluster: str = 'MPCDF',
               namelistFile: str = None, fileOutput: bool = False):
    """
    Execute a SINPA simulation

    Jose Rueda: jrrueda@us.es

    :param  runID: runID of the simulation
    :param  namelist: full path to the namelist
    :param  queue: Flag to launch the FILDSIM simulation into the queue
    :param  cluster: string identifying the cluster. Each cluster may require
        different submition option. Up to now, only MPCDF ones are supported
    :param  namelistFile: full path to the namelist file. If None, it will
        look for the namelist in the inputs folder of the runID
    :param  fileOutput: If True, the output will be written to a file, else it
        will be printed to the terminal, only working for no queue mode
    """
    if not queue:       # Just execute the code in the current terminal
        SINPAbinary = os.path.join(paths.SINPA, 'bin', 'SINPA.go')
        if namelistFile is None:
            p = os.path.join(paths.SINPA, 'runs', runID, 'inputs',
                             runID + '.cfg')
        else:
            p = namelistFile
        if fileOutput:
            # If fileOutput is True, we will write the output to a file
            folder, file = os.path.split(p)
            extra = ' > ' + os.path.join(folder, 'out.txt')
        else:
            extra = ''
        os.system(SINPAbinary + ' ' + p + extra)
    else:  # Prepare a simulation to be launched in the queue
        if cluster.lower() == 'mpcdf':
            p = os.path.join(paths.SINPA, 'runs', runID,
                             'inputs', runID + '.cfg')
            result_dir = os.path.join(paths.SINPA, 'runs', runID, 'results')
            nml = reading.read_namelist(p)
            f = open(result_dir+'/Submit.sh', 'w')
            f.write('#!/bin/bash -l \n')
            f.write('#SBATCH -J FILDSIM_%s      #Job name \n' %
                    (nml['config']['runid']))
            f.write('#SBATCH -o ./%x.%j.out        '
                    + '#stdout (%x=jobname, %j=jobid) \n')
            f.write('#SBATCH -e ./%x.%j.err        '
                    + '#stderr (%x=jobname, %j=jobid) \n')
            f.write('#SBATCH -D ./              #Initial working directory \n')
            f.write('#SBATCH --partition=s.tok     #Queue/Partition \n')
            f.write('#SBATCH --qos=s.tok.short \n')
            f.write('#SBATCH --nodes=1             #Total number of nodes \n')
            f.write('#SBATCH --ntasks-per-node=1   #MPI tasks per node \n')
            f.write('#SBATCH --cpus-per-task=1   #CPUs per task for OpenMP \n')
            f.write('#SBATCH --mem 5GB          #Set mem./node requirement \n')
            f.write('#SBATCH --time=03:59:00       #Wall clock limit \n')
            f.write('## \n')
            f.write('#SBATCH --mail-type=end       #Send mail\n')
            f.write('#SBATCH --mail-user=%s@ipp.mpg.de  #Mail address \n'
                    % (os.getenv("USER")))

            f.write('# Run the program: \n')
            SINPAbinary = os.path.join(paths.SINPA, 'bin', 'SINPA.go')
            f.write(SINPAbinary + ' ' + p)
            f.close()

            os.system('sbatch ' + result_dir + '/Submit.sh')
        else:
            raise errors.NotImplementedError()
    return
