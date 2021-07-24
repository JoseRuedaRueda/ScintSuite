"""
i-HIBP namelists and scans generator.

This series of routines will be useful to generate a namelist or a series of
namelists ready to simulate with some of the i-HIBPsim codes. 

The meaning of each of the parameters is explained in the i-HIBPsim repository.

https://gitlab.mpcdf.mpg.de/poyo/ihibpsim/

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import warnings
import Lib

try:
    import f90nml
except ImportError:
    warnings.warn('You cannot read FILDSIM namelist nor remap',
                  category=UserWarning)


def make_tracker_namelist(user_nml: dict):
    """
    Write fortran namelist

    Jose rueda: jrrueda@us.es

    Just a wrapper for the f90nml file writter

    To see the meaning of all parameters, look at the nicely written iHIBPsim
    documentation

    @param user_nml: namelist containing the desired fields.
    @param file: full path towards the desired file
    @param overwrite: flag to overwrite the namelist (if exist)

    f90nml format adopted in version 0.4.10
    """
    # Default namelist:
    nml = {
        'FIELD_FILES': {
            'Bfield_name': '',
            'Efield_name': '',
            'Efield_on': False,
            'equ_file': '',
        },
        'PROFILES': {
            'Te_name': '',
            'ne_name': '',
            'n0_name': '',
            'Zeff': 1.0
        },
        'TABLES': {
            'beamAttenuationModule': False,
            'elec_name': '',
            'CX_name': '',
        },
        'INTEGRATION': {
            'dt': 1.0e-9,
            'max_step': 20000000,
            'file_out': 'strikes.bin',
        },
        'ORBITS_CONF': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'orbit.bin',
            'dt_orbit': 1.0e-8,
        },
        'DEPOSITION': {
            'markerNumber': 1,
            'depos_file': 'markers.bin',
            'verbose': True
        },
        'SCINTILLATOR': {
            'triangle_file': 'scintillator.dat'
        }
    }
    # Update the fields, if we just use nml.update(user_nml), if user_nml has
    # the block 'ORBITS_CONF', but inside it just the field 'save_orbits',
    # because the user only wants to update that field, it will fail, as all
    # the block 'ORBITS_CONF' would be replaced by one with just that field, so
    # we need to perform the comparison one by one
    for key in nml.keys():
        if key in user_nml:
            nml[key].update(user_nml[key])
    
    return nml


def make_ihibpsim1_namelist(user_nml: dict):
    """
    Write the namelist for the iHIBPsim simulation. i-HIBPsim is the program
    able to compute the initial deposition as well as the scintillator
    striking points.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    A wrapper to write and generate a valid namelist to use with the 
    i-HIBPsim code.
    
    @param user_nml: namelist containing the desired fields.
    """
    # Default namelist:
    nml = {
        'FIELD_FILES': {
            'Bfield_name': '',
            'Efield_name': '',
            'Efield_on': False,
            'equ_file': '',
        },
        'PROFILES': {
            'beamAttflag': False,
            'Te_name': '',
            'ne_name': '',
            'n0_name': '',
            'Zeff': 1.0
        },
        'TABLES': {
            'elec_name': '',
            'CX_name': '',
            'prm_name': ''
        },
        'INTEGRATION': {
            'dt': 1.0e-11,
            'max_step': 100000,
            'file_out': 'strikes.bin',
            'verbose': True
        },
        'ORBITS_CONF': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'orbit.bin',
            'dt_orbit': 1.0e-8,
        },
        'DEPOSITION': {
            'NbeamDir': 128,
            'Ndisk': 1,
            'Rmin': 1.75,
            'Rmax': 2.20,
            'depos_file': 'markers.bin',
        },
        'SCINTILLATOR': {
            'triangle_file': 'scintillator.dat'
        }
    }
    # Update the fields, if we just use nml.update(user_nml), if user_nml has
    # the block 'ORBITS_CONF', but inside it just the field 'save_orbits',
    # because the user only wants to update that field, it will fail, as all
    # the block 'ORBITS_CONF' would be replaced by one with just that field, so
    # we need to perform the comparison one by one
    for key in nml.keys():
        if key in user_nml:
            nml[key].update(user_nml[key])
    
    return nml

def make_shotmapper_namelist(user_nml: dict):
    """
    Write the namelist for the iHIBPsim simulation. This routine is used to
    generate the strikeline and scintillator image for i-HIBPsim diagnostic.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    A wrapper to write and generate a valid namelist to use with the 
    shot_mapper code.
    
    @param user_nml: namelist containing the desired fields.
    """
    # Default namelist:
    nml = {
        'SHOT': {
            'shotnumber': 0,     # Setting this will retrieve the last shot.
            'expMagn': 'AUGD',
            'diagMagn': 'EQI',
            'expProf': 'AUGD',
            'diagProf': 'IDA',
            'timemin': 0.10,
            'timemax': 10.0,
            'dt_shot': 0.50,
            'limits': np.array((1.65, 2.65, -1.0, 1.0)),
            'Nr': 512,
            'Nz': 256
        },
        'TABLES': {
            'beamAttenuationFlag': False,
            'elec_name': '',
            'CX_name': '',
            'prm_name': '',
            'Zeff1': 1.0e0
        },
        'INTEGRATION': {
            'dt': 1.0e-11,
            'max_step': 100000,
            'verbose': True
        },
        'ORBITS_CONF': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'orbits',
            'dt_orbit': 1.0e-8,
        },
        'DEPOSITION': {
            'NbeamDir': 128,
            'Ndisk': 1,
            'storeDeposition': True,
        },
        'SCINT3D': {
            'triangle_file': 'scintillator.dat'
        },
        
        'STRIKEMAPCONF': {
            'scint_vertex_file': 'scintmapfile.dat',
            'mode': 2,
            'startR': 0.85,
            'endR': 1.20,
            'strikemap_file': 'strikemap.map',
            'save_striking_points': False,
            'file_strikes': 'strikes'
        },
        
        'GEOMETRY': {
            'origin_point': Lib.dat.iHIBP['port_center'],
            'tilting_beta': Lib.dat.iHIBP['beta_std'],
            'tilting_theta': Lib.dat.iHIBP['theta_std'],
            'beamModelOrder': 0,
            'radius': Lib.dat.iHIBP['source_radius'],
            'sourceRadius': Lib.dat.iHIBP['source_radius'],
            'divergency': 0.0,
            'mean_energy': 50.0,
            'std_E': 0.0,
            'mass': 87.0,
            'intensity': 1.0e-3
            
        }
    }
    # Update the fields, if we just use nml.update(user_nml), if user_nml has
    # the block 'ORBITS_CONF', but inside it just the field 'save_orbits',
    # because the user only wants to update that field, it will fail, as all
    # the block 'ORBITS_CONF' would be replaced by one with just that field, so
    # we need to perform the comparison one by one
    for key in nml.keys():
        if key in user_nml:
            nml[key].update(user_nml[key])
    
    return nml


def generate_energy_scan_shotmapper(E_start: float, E_end: float, nE: int,
                                    base_nml: dict={}, code='shotmapper'):
    
    """
    This routine will generate a set of dictionaries containing the data for
    performing an energy scan.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param E_start: lowest energy to try.
    @param E_end: highest energy to try.
    @param nE: number of points of the analysis.
    @param base_nml: this is basic namelist to use provided by the user.
    """
    
    if E_start <= 0 or E_end <= 0:
        raise ValueError('The particle energy cannot be negative!')
        
    energy = np.linspace(E_start, E_end, nE)
    
    nml_list = {}
    
    for ienergy, E in enumerate(energy):
        if code == 'shotmapper':
            nml_list[E] = make_shotmapper_namelist(base_nml)
        elif code == 'ihibpsim':
            nml_list[E] = make_ihibpsim1_namelist(base_nml)
            
        nml_list[E]['GEOMETRY']['energy'] = E
        
    return nml_list

def generate_beta_scan_shotmapper(beta_start: float, beta_end: float,
                                  nbeta: int, base_nml: dict={}, 
                                  code='shotmapper'):
    
    """
    This routine will generate a set of dictionaries containing the data for
    performing an energy scan.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param E_start: lowest energy to try.
    @param E_end: highest energy to try.
    @param nE: number of points of the analysis.
    @param base_nml: this is basic namelist to use provided by the user.
    """
    
    if (abs(beta_start) > np.pi/2) or (abs(beta_end) > np.pi/2):
        raise ValueError('Pitch angle cannot be larger than pi/2')
        
    beta = np.linspace(beta_start, beta_end, nbeta)
    
    nml_list = {}
    
    for ibeta, bet in enumerate(beta):
        if code == 'shotmapper':
            nml_list[bet] = make_shotmapper_namelist(base_nml)
        elif code == 'ihibpsim':
            nml_list[bet] = make_ihibpsim1_namelist(base_nml)
            
        nml_list[bet]['GEOMETRY']['tilting_beta'] = bet
        
    return nml_list