"""
Read data from SINPA

Introduced in version 6.0.0

Jose Rueda Rueda: jrrueda@us.es
"""
import f90nml


def read_namelist(filename, verbose=False):
    """
    Read a FILDSIM namelist

    Jose Rueda: jrrueda@us.es

    just a wrapper for the f90nml capabilities, copied from the FILDSIM library

    @param filename: full path to the filename to read
    @param verbose: Flag to print the nml in the console

    @return nml: dictionary with all the parameters of the FILDSIM run
    """
    nml = f90nml.read(filename)
    if verbose:
        print(nml)
    return nml
