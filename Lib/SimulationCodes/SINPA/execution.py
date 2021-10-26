"""
Contains the routines to easily execute the SINPA code

Jose Rueda: jrrueda@us.es

Introduced in version 6.0.0
"""
import os
import f90nml
import numpy as np
import Lib.SimulationCodes.SINPA.reading as reading
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)


def write_namelist(nml, p=None, overwrite=True):
    """
    Write fortran namelist

    jose rueda: jrrueda@us.es

    Just a wrapper for the f90nml file writter. It differ from the one of the
    fildsim package in the folder structure used to save the data.

    @param nml: namelist containing the desired fields.
    @param p: full path towards the root folder of SINPA. In principle it will
    take it from the path of the suite. Please do not use this input except you
    really know what are you doing and want to change something
    @param overwrite: flag to overwrite the namelist (if exist)

    @return file: The path to the written file
    """
    # Initialise the path
    if p is None:
        p = os.path.join(paths.SINPA, 'runs', nml['config']['runid'], 'inputs')
    # Get the full name of the namelist to be saved
    file = os.path.join(p, nml['config']['runid'] + '.cfg')
    f90nml.write(nml, file, force=overwrite)
    return file


def check_files(runID: str):
    """
    Check that the necesary files to execute SINPA exist

    Jose Rueda Rueda: jrrueda@us.es

    Note that this function will not check the files itself, just it will look
    that the files exist

    @param runID: RunID of the simulation

    @return go: Flag to say if we are ready to launch SINPA
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


def executeRun(runID: str, queue: bool = False):
    """
    Execute a SINPA simulation

    Jose Rueda: jrrueda@us.es

    Note, the queue part will preare the bach o be compatible with IPP cluster
    and will send the email to the IPP email adress, so that option is not
    machine independent

    @param runID: runID of the simulation
    """
    if not queue:       # Just execute the code in the current terminal
        SINPAbinary = os.path.join(paths.SINPA, 'bin', 'SINPA.go')
        p = os.path.join(paths.SINPA, 'runs', runID, 'inputs', runID + '.cfg')
        os.system(SINPAbinary + ' ' + p)
    else:               # Prepare a simulation to be launched in the queue
        p = os.path.join(paths.SINPA, 'runs', runID, 'inputs', runID + '.cfg')
        nml = reading.read_namelist(p)
        f = open(nml['config']['result_dir']+'/Submit.sh', 'w')
        f.write('#!/bin/bash -l \n')
        f.write('#SBATCH -J FILDSIM_%s      #Job name \n' %
                (nml['config']['runid']))
        f.write('#SBATCH -o ./%x.%j.out        '
                + '#stdout (%x=jobname, %j=jobid) \n')
        f.write('#SBATCH -e ./%x.%j.err        '
                + '#stderr (%x=jobname, %j=jobid) \n')
        f.write('#SBATCH -D ./                 #Initial working directory \n')
        f.write('#SBATCH --partition=s.tok     #Queue/Partition \n')
        f.write('#SBATCH --qos=s.tok.short \n')
        f.write('#SBATCH --nodes=1             #Total number of nodes \n')
        f.write('#SBATCH --ntasks-per-node=1   #MPI tasks per node \n')
        f.write('#SBATCH --cpus-per-task=1     #CPUs per task for OpenMP \n')
        f.write('#SBATCH --mem 5GB             #Set mem./node requirement \n')
        f.write('#SBATCH --time=03:59:00       #Wall clock limit \n')
        f.write('## \n')
        f.write('#SBATCH --mail-type=end       #Send mail\n')
        f.write('#SBATCH --mail-user=%s@ipp.mpg.de  #Mail address \n'
                % (os.getenv("USER")))

        f.write('# Run the program: \n')
        SINPAbinary = os.path.join(paths.SINPA, 'bin', 'SINPA.go')
        f.write(SINPAbinary + ' ' + p)
        f.close()

        os.system('sbatch ' + nml['config']['result_dir'] + '/Submit.sh')
    return
