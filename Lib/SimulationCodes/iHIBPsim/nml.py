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
import os
from Lib.LibMachine import machine
from Lib.LibPaths import Path

try:
    import f90nml
except ImportError:
    warnings.warn('You cannot read FILDSIM namelist nor remap',
                  category=UserWarning)
    
IHIBPSIM_ACTION_NAMES = ('tracker', 'ihibpsim', 'shot_remap')
paths = Path(machine)


#-----------------------------------------------------------------------------
# Routines to make namelists to run iHIBPsim.
#-----------------------------------------------------------------------------
def make_namelist(codename: str, user_nml: dict):
    """
    Wrapper to call the generators of namelists contained in this library.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param codename: name of the code to run. @see IHIBPSIM_ACTION_NAMES.
    @param user_nml: user provided namelist.
    """
    
    if codename == 'tracker':
        return make_tracker_namelist(user_nml)
    elif codename == 'ihibpsim':
        return make_ihibpsim1_namelist(user_nml)
    elif codename == 'shot_mapper':
        return make_shotmapper_namelist(user_nml)


def make_tracker_namelist(user_nml: dict):
    """
    Write fortran namelist

    Jose rueda: jrrueda@us.es

    Just a wrapper for the f90nml file writter

    To see the meaning of all parameters, look at the nicely written iHIBPsim
    documentation

    @param user_nml: namelist containing the desired fields.
    @param path: destiny path of the results.

    f90nml format adopted in version 0.4.10
    """
    # Default namelist:
    nml = {
        'field_files': {
            'bfield_name': '',
            'efield_name': '',
            'efield_on': False,
            'equ_file': '',
        },
        'profiles': {
            'te_name': '',
            'ne_name': '',
            'n0_name': '',
            'zeff': 1.0
        },
        'tables': {
            'beamattenuationmodule': False,
            'elec_name': '',
            'cx_name': '',
        },
        'integration': {
            'dt': 1.0e-9,
            'max_step': 20000000,
            'file_out': 'strikes.bin',
        },
        'orbits_conf': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'orbit.bin',
            'dt_orbit': 1.0e-8,
        },
        'deposition': {
            'markernumber': 1,
            'depos_file': 'markers.bin',
            'verbose': True
        },
        'scintillator': {
            'triangle_file':'scintillator.dat'
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
        'field_files': {
            'bfield_name': '',
            'efield_name': '',
            'efield_on': False,
            'equ_file': '',
        },
        'profiles': {
            'te_name': '',
            'ne_name': '',
            'n0_name': '',
            'zeff': 1.0
        },
        'tables': {
            'beamattenuationmodule': False,
            'elec_name': '',
            'cx_name': '',
            'prm_name': ''
        },
        'integration': {
            'dt': 1.0e-11,
            'max_step': 100000,
            'file_out': 'output.strikes',
            'verbose': True
        },
        'orbits_conf': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'output.orbits',
            'dt_orbit': 1.0e-8,
        },
        'deposition': {
            'nbeamdir': 128,
            'ndisk': 1,
            'rmin': 1.75,
            'rmax': 2.20,
            'depos_file': 'output.beam',
        },
        
        'geometry': {
            'origin_point': Lib.dat.iHIBP['port_center'],
            'tilting_beta': Lib.dat.iHIBP['beta_std'],
            'tilting_theta': Lib.dat.iHIBP['theta_std'],
            'beammodelorder': 0,
            'radius': Lib.dat.iHIBP['source_radius'],
            'sourceradius': Lib.dat.iHIBP['source_radius'],
            'divergency': 0.0,
            'mean_energy': 50.0,
            'std_e': 0.0,
            'mass': 87.0,
            'intensity': 1.0e-3
            
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
        'shot': {
            'shotnumber': 0,     # Setting this will retrieve the last shot.
            'expmagn': 'AUGD',
            'diagmagn': 'EQI',
            'expprof': 'AUGD',
            'diagprof': 'IDA',
            'timemin': 0.10,
            'timemax': 10.0,
            'dt_shot': 0.50,
            'limits': np.array((1.65, 2.65, -1.0, 1.0)),
            'nr': 512,
            'nz': 256
        },
        'tables': {
            'beamattenuationflag': False,
            'elec_name': '',
            'cx_name': '',
            'prm_name': '',
            'zeff1': 1.0e0
        },
        'integration': {
            'dt': 1.0e-11,
            'max_step': 100000,
            'verbose': True
        },
        'orbits_conf': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'orbits',
            'dt_orbit': 1.0e-8,
        },
        'deposition': {
            'nbeamDir': 128,
            'ndisk': 1,
            'storedeposition': True,
        },
        'scint3d': {
            'triangle_file': 'scintillator.3d'
        },
        
        'strikemapconf': {
            'scint_vertex_file': 'scintmapfile.dat',
            'mode': 2,
            'startr': 0.85,
            'endr': 1.20,
            'strikemap_file': 'strikemap.map',
            'save_striking_points': False,
            'file_strikes': 'strikes'
        },
        
        'geometry': {
            'origin_point': Lib.dat.iHIBP['port_center'],
            'tilting_beta': Lib.dat.iHIBP['beta_std'],
            'tilting_theta': Lib.dat.iHIBP['theta_std'],
            'beammodelrrder': 0,
            'radius': Lib.dat.iHIBP['source_radius'],
            'sourceradius': Lib.dat.iHIBP['source_radius'],
            'divergency': 0.0,
            'mean_energy': 50.0,
            'std_e': 0.0,
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

#-----------------------------------------------------------------------------
# Routines to generate scans.
#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
# Routines to check the namelist consistency.
#-----------------------------------------------------------------------------
def check_namelist(params:dict, codename: str='ihibpsim',
                   forceConvention: bool=True):
    """
    Wrapper around the internal routines that checks whether the data inside
    the dictionary "params" would appropriately correspond to a proper made
    namelist.
    WARNING: This does not check whether the files exists on the folder or not.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param params: dictionary with the namelist to run iHIBPsim.
    @param codename: name of the executable to run. To choose among:
        ('tracker', 'ihibpsim', 'shot_mapper')
    """
    
    if codename == 'tracker':
        return __check_tracker_namelist(params, forceConvention)
    elif codename == 'ihibpsim':
        return __check_ihibpsim_namelist(params, forceConvention)
    elif codename == 'shot_mapper':
        return __check_shotmapper_namelist(params, forceConvention)
    else:
        raise ValueError('The code %s is not a valid code name'%codename)
       
        
def __check_tracker_namelist(params: dict, forceConvention: bool=True):
    """
    Internal routine to check the consistency of a namelist to run the tracker
    built with the iHIBPsim libraries.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param params: dictionary containing the namelist.
    @param forceConvention: this flag will ensure that all files end with 
    an appropriate extension that can be easily recognized by this suite.
    """
    
    # Returns an error code corresponding to the error while parsing the 
    # namelist.
    
    # Checks that all the elements in the namelist names are in the dictionary.
    minimal_names =  ('field_files', 'profiles', 'tables', 'integration',
                      'orbits_conf', 'deposition', 'scintillator')
    
    if np.any([ii not in params for ii in minimal_names]):
        return (-1, 'Namelist not proper structure')
    
    # Checking the field files data:
    if params['field_files']['bfield_name'] == '':
        return (-2, 'Magnetic field filename is empty')
    else:
        if forceConvention and\
            (not params['field_files']['bfield_name'].endswith('.bfield')):
                return (-3, 'ForceConvention: The magnetic field must have '+\
                        'extension .bfield')
    
    
    if (params['field_files']['efield_on']):
        if (params['field_files']['efield_name'] == ''):
            return (-4, 'Electric field filename is empty and flag is set!')
        else:
            if forceConvention and\
                (not params['field_files']['efield_name'].endswith('.efield')):
                return (-5, 'ForceConvention: The electric field must have '+\
                        'extension .efield')
    
    # Checking if the profiles are set.
    if params['tables']['beamattenuationmodule']:
        if (params['tables']['elec_name'] == '') and\
           (params['tables']['CX_name'] == ''):
               warnings.warn('The beam attenuation module is set, but none '+\
                             'of the tables are loaded. Disabling...',
                             category=UserWarning)
                  
               params['tables']['beamattenuationmodule'] = False
    
    if params['tables']['beamattenuationmodule']:
        if (params['profiles']['te_name'] == ''):
            return (-6, 'Beam attenuation module is set, but not Te given')
        else:
            if forceConvention and\
                (not params['profiles']['te_name'].endswith('.te')):
                return (-7, 'ForceConvention: The electron density must have '+\
                        'extension .te')
                    
        if (params['profiles']['ne_name'] == ''):
            return (-8, 'Beam attenuation module is set, but not ne given')
        else:
            if forceConvention and\
                (not params['profiles']['ne_name'].endswith('.ne')):
                return (-9, 'ForceConvention: The electron density must have '+\
                        'extension .ne')
        
        if (params['profiles']['n0_name'] == ''):
            print('The n0 species will be considered to be'+\
                  ' equal to the electrons')
                
        if (params['profiles']['Zeff'] <= 0):
            return (-10, 'Only Zeff > 0 is possible!')
        
        
    # Checking the integration time step and outputs.
    if params['integration']['dt'] < 0:
        return (-11, 'Integration time step has to be larger than 0')
    
    
    if params['integration']['max_step'] < 1:
        return (-12, 'The number of steps has be 1 or larger')
    
    
    if params['integration']['file_out'] == '':
        return (-13, 'The output filename must be provided!')
    else:
        if forceConvention and\
            (not params['integration']['file_out'].endswith('.strikes')):
            return (-14, 'ForceConvention: output file must end with the '+
                    'extension .strikes')
    
    # Checking the orbits configuration.
    if params['orbits_conf']['save_orbits']:
        params['orbits_conf']['num_orbits'] = min(1.0, 
                                              params['orbits_conf']\
                                                    ['num_orbits'])
        if params['orbits_conf']['num_orbits'] <= 0:
            warnings.warn('The number of orbits to store is set to 0 by the '+\
                          'user. Setting the orbits flag to False.',
                          category=UserWarning)
            
            params['orbits_conf']['save_orbit'] = False
        else:
            if params['orbits_conf']['file_orbits'] == '':
                return(-15, 'The orbits module is set, '+\
                       'but the filename is empty')
            
            else:
                if forceConvention and\
                  (not params['orbits_conf']['file_orbits'].endswith('.orbits')):
                    return (-16, 'ForceConvention: the orbit file must '+\
                            'end with the extension .orbits')
            
            # Checking the time integration for the orbit storage.
            if params['orbits_conf']['dt_orbit'] < params['integration']['dt']:
                params['orbits_conf']['dt_orbit'] = params['integration']['dt']
                warnings.warn('The time step for orbit saving is smaller'+\
                              ' than the integration time. Setting them equal',
                              category=UserWarning)
    
        # Checking the deposition module input.
        if params['deposition']['markerNumber'] < 1:
            return (-17, 'The number of markers must be 1 or larger')
        
        if params['depostion']['depos_file'] == '':
            return (-18, 'The file with the deposition must be provided')
        else:
            if forceConvention and\
                  (not params['depostion']['depos_file'].endswith('.markers')):
                    return (-19, 'ForceConvention: the initial particle file'+\
                            ' must end with the extension .markers')
                        
        # Cheking that there is a triangle file to stop the ions.
        if params['scintillator']['triangle_file'] == '':
            return (-20, 'A file containing the triangularization of the '+
                    'stopping points of the ions must be provided!')
        else:
            if forceConvention and\
                  (not params['depostion']['depos_file'].endswith('.3d')):
                    return (-19, 'ForceConvention: the scintillator file'+\
                            ' must end with the extension .3d')


def  __check_ihipbsim_geometry(geom: dict):
    """
    Checks the consistency of the geometry namelist for iHIBPsim.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param geom: dictionary containing the parameters of the beam geometry.
    """
    
    minimal_names = ('tilting_beta', 'tilting_theta', 'origin_point', 
                     'beammodelorder', 'radius', 'sourceradius',
                     'divergency', 'mean_energy', 'std_e', 'mass',
                     'intensity')
    if np.any([ii not in geom for ii in minimal_names]):
        return (-26, 'The geometry module does not contain all the needed data')
    
    
    if abs(geom['tilting_beta']) >= 90.0:
        return (-27, 'The toroidal tilt cannot be larger than 90 degrees')
    if abs(geom['tilting_theta']) >= 90.0:
        return (-28, 'The poloidal tilt cannot be larger than 90 degrees')
    
    if geom['beammodelorder'] not in (0, 1, 2, 3):
        return (-29, 'The beam mode #%d is not recognized'%geom['beammodelorder'])
    
    if geom['radius'] < 0.0:
        return (-30, 'The beam radius cannot be negative [m]')
    if geom['divergency'] < 0.0:
        return (-31, 'The divergency cannot be negative [º]')
    if geom['std_e'] < 0.0:
        return (-32, 'The energy spread cannot be negative [keV]')
    if geom['sourceradius'] <= 0.0:
        return (-33, 'The source radius cannot be negative [m]')
    
    if geom['beammodelorder'] == 1:
        if geom['radius'] == 0.0:
            warnings.warn('Setting model for beam to 0: no beam width')
            geom['beammodelorder'] = 0
    
    elif geom['beammodelorder'] == 2:
        if (geom['divergency'] <= 0.0) and (geom['radius'] == 0.0):
            warnings.warn('Setting model to 0: no beam width nor divergency!')
            geom['beammodelorder'] = 0
        elif geom['divergency'] <= 0.0:
            warnings.warn('The divergency is set to 0 but the'+\
                          ' model includes it. Setting model to 1.')
            geom['beammodelorder'] = 1
    elif geom['beammodelorder'] == 3:
        if (geom['divergency'] <= 0.0) and (geom['radius'] == 0.0) \
            and (geom['std_e'] <= 0.0):
            warnings.warn('Setting model to 0: no finite effects!')
            geom['beammodelorder'] = 0
        if (geom['divergency'] <= 0.0) and (geom['std_e'] <= 0.0):
            warnings.warn('The divergencies is set to 0 but the'+\
                          ' model includes it. Setting model to 1.')
            geom['beammodelorder'] = 1
        elif (geom['std_e'] <= 0.0):
            warnings.warn('The energy spread is set to 0, but the model '+\
                          'includes it. Setting model order = 2')
            geom['beammodelorder'] = 2
            
    # Checking the beam species parameter:
    if geom['energy'] <= 0.0:
        return (-34, 'Beam energy must be positive [in keV]')
    
    if geom['mass'] <= 0.0:
        return (-35, 'Beam-particle mass must be positive [in amu]')
    
    if geom['intensity'] <= 0.0:
        return (-35, 'Beam-particle mass must be positive [in A]')
    
    if len(geom['origin_point']) != 3:
        return (-36, 'The origin point must be 3 cartesian coordinates!')


def __check_ihibpsim_namelist(params: dict, forceConvention: bool=True):
    """
    Internal routine that checks the consistency of the namelist before
    launching a simulation with the iHIPBsim tracker.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param params: namelist as a dictionary to be checked.
    @param forceConvention: this flag will ensure that all files end with 
    an appropriate extension that can be easily recognized by this suite.
    """
    
    # The first part of the namelist for tracker and the ihibpsim code
    # are identical.
    err, msg = __check_tracker_namelist(params=params,
                                        forceConvention=forceConvention)
    if (err != 0) and (err  != -17):
        return err, msg
    
    # Checking the particular modules of ihibpsim:
    # 1. The deposition.
    nparticles = params['deposition']['Nbeamdir'] *\
                 params['deposition']['Ndisk']
                 
    if nparticles < 1:
        return (-22, 'The number of particles to launch in iHIBPsim must be '+\
                'at least 1.')
    
    if params['deposition']['Rmin'] <= 0.0:
        return (-23, 'The deposition minimum radius must be positive')
    
    if params['deposition']['Rmin'] <= 0.0:
        return (-24, 'The deposition maximum radius must be positive')
    
    if params['deposition']['Rmin'] > params['deposition']['Rmax']:
        warnings.warn('Min/max values of the deposition radii seem to be '+\
                      'exchanged. Setting the appropriate order.')
        
        tmp = params['deposition']['Rmin']
        params['deposition']['Rmin'] = params['deposition']['Rmax']
        params['deposition']['Rmax'] = tmp
    
    if params['deposition']['Rmin'] == params['deposition']['Rmax']:
        if params['deposition']['Nbeamdir'] != 0:
            return (-25, 'Rmin=Rmax, but more than one point along'+\
                    ' the injection line are considered')
    
    # Checking the geometry part of the namelist.
    return __check_ihipbsim_geometry(params['geometry'])


def __check_shotmapper_namelist(params: dict, forceConvention: bool=True):
    """
    Internal routine to check the namelist to run the shotmapper namelist
    prior to a run.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param params: namelist as a dictionary to be checked.
    @param forceConvention: this flag will ensure that all files end with 
    an appropriate extension that can be easily recognized by this suite.
    """
    
    # Checking that the namelist has the elements neccessary.
    minimal_names = ('shot', 'tables', 'integration', 'orbits_conf',
                     'deposition', 'scit3d', 'strikemapconf', 'geometry')
    
    if np.any([ii not in params for ii in minimal_names]):
        return (-37, 'Namelist uncomplete for shot mapping binary.')
    
    # 1. Checking the shot data.
    if (params['shot']['shotnumber'] < 0) and\
       (params['shot']['shotnumber'] < 100000):
           return (-38, 'The shotnumber must be positive and in range')
    
    if len(params['shot']['expMagn']) > 10:
        return (-39, 'Experiment name for magnetics cannot be that large')
    if len(params['shot']['diagMagn']) > 3:
        return (-40, 'Diagnostic name for magnetics cannot be that large')
    if len(params['shot']['expProf']) > 10:
        return (-41, 'Experiment name for profiles cannot be that large')
    if len(params['shot']['diagProf']) > 3:
        return (-42, 'Diagnostic name for profiles cannot be that large')
        
    if params['shot']['timemin'] > params['shot']['timemax']:
        warnings.warn('The shot min/max times are exchanged! Reversing them',
                      category=UserWarning)
        tmp = params['shot']['timemin']
        params['shot']['timemin'] = params['shot']['timemax']
        params['shot']['timemax'] = tmp
    
    dtparse = params['shot']['timemax'] - params['shot']['timemin']
    if params['shot']['dt_shot'] > dtparse:
        warnings.warn('The time step for parsing is larger '+\
                      'than the time window. Adjusting the time to the limits',
                      category=UserWarning)
            
        params['shot']['dt_shot'] = dtparse
    
    if params['shot']['nr'] < 1:
        return (-43, 'The number of radial points must be positive!')
    if params['shot']['nz'] < 1:
        return (-44, 'The number of vertical points must be positive!')
    
    # Checking the grid limits.
    rmin, rmax, zmin, zmax = params['shot']['limits']
    if rmin > rmax:
        warnings.warn('The grid radial limits min/max times are exchanged!'+\
                      'Reversing them', category=UserWarning)
        
        tmp = rmin
        rmin = rmax
        rmax = tmp
        
    if zmin > zmax:
        warnings.warn('The grid vertical limits min/max times are exchanged!'+\
                      'Reversing them', category=UserWarning)
        
        tmp = zmin
        zmin = zmax
        zmax = tmp
        
    if (rmin < 0) or (rmax < 0):
        return (-45, 'The limits in the radial direction cannot be negative')

    params['shot']['limits'] = (rmin, rmax, zmin, zmax)
    
    # 2. Checking the ionization tables.
    if params['tables']['beamattenuationmodule']:
        if (params['tables']['elec_name'] == '') and\
           (params['tables']['CX_name'] == '') and\
           (params['tables']['prm_name'] == ''):
               warnings.warn('The beam attenuation module is set, but none '+\
                             'of the tables are loaded. Disabling...',
                             category=UserWarning)
                  
               params['tables']['beamattenuationmodule'] = False
    
    # 3. checking the integration step and the orbits configuration.
    if params['integration']['dt'] < 0:
        return (-46, 'Integration time step has to be larger than 0')
    
    
    if params['integration']['max_step'] < 1:
        return (-47, 'The number of steps has be 1 or larger')
    
    # Checking the orbits configuration.
    if params['orbits_conf']['save_orbits']:
        params['orbits_conf']['num_orbits'] = min(1.0, 
                                              params['orbits_conf']\
                                                    ['num_orbits'])
        if params['orbits_conf']['num_orbits'] <= 0:
            warnings.warn('The number of orbits to store is set to 0 by the '+\
                          'user. Setting the orbits flag to False.',
                          category=UserWarning)
            
            params['orbits_conf']['save_orbit'] = False
        else:
            if params['orbits_conf']['file_orbits'] == '':
                return(-48, 'The orbits module is set, '+\
                       'but the filename is empty')
            
            # Checking the time integration for the orbit storage.
            if params['orbits_conf']['dt_orbit'] < params['integration']['dt']:
                params['orbits_conf']['dt_orbit'] = params['integration']['dt']
                warnings.warn('The time step for orbit saving is smaller'+\
                              ' than the integration time. Setting them equal',
                              category=UserWarning)
    
    # 4. Checking the beam deposition profile.
    nparticles = params['deposition']['Nbeamdir'] *\
                 params['deposition']['Ndisk']
                 
    if nparticles < 1:
        return (-49, 'The number of particles to launch in iHIBPsim must be '+\
                'at least 1.')
            
    # 5. Checking the scintillator triangle file.
    if params['scint3d']['triangle_file'] == '':
        return (-50, 'A file containing the triangularization of the '+
                'stopping points of the ions must be provided!')
    else:
        if forceConvention and\
              (not params['depostion']['depos_file'].endswith('.3d')):
                return (-51, 'ForceConvention: the scintillator file'+\
                            ' must end with the extension .3d')
                    
    # 6. Checking the strike map configuration.
    if params['strikemapconf']['scint_vertex_file'] == '':
        return (-52, 'The scintillator file cannot be empty.')
    
    if params['strikemapconf']['mode'] not in (0, 1, 2):
        return (-53, 'The mode number must be 0, 1, 2')
    
    if params['strikemapconf']['startR'] < 0:
        return (-54, 'Initial mapping point must be positive')
    
    if params['strikemapconf']['endR'] < 0:
        return (-55, 'Ending mapping point must be positive')
    
    if params['strikemapconf']['startR'] > params['strikemapconf']['endR']:
        warnings.warn('The mapped striked points min/max times '+\
                       'are exchanged! Reversing them', category=UserWarning)
             
        tmp = params['strikemapconf']['startR']
        params['strikemapconf']['startR'] = params['strikemapconf']['endR']
        params['strikemapconf']['endR']   = tmp
    
    if params['strikemapconf']['save_striking_points']:
        if params['strikemapconf']['file_strikes'] == '':
            return (-56, 'The filename of the strike files must be filled!')
            
        
    # 7. Check the geometry.
    return __check_ihipbsim_geometry(params['geometry'])

#-----------------------------------------------------------------------------
# Routines to check the files as set in the namelist
#-----------------------------------------------------------------------------
def check_files(nml: dict, action: str='shot_remap'):
    """
    Check if the needed files for the execution of the codes are properly
    stored in the corresponding folders.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param params: dictionary with the namelist to run iHIBPsim.
    @param action: code to run. To be chosen among IHIBPSIM_ACTION_NAMES
    """
    flags = True
    # For each code, we need to verify that the 
    # corresponding files are there.
    if action == 'tracker':
        flags &= os.path.isfile(nml['field_files']['bfield_name'])
        if nml['field_files']['efield_on']:
            flags &= os.path.isfile(nml['field_files']['efield_name'])
        if nml['tables']['beamattenuationmodule']:
            flags &= os.path.isfile(nml['field_files']['te_name'])
            flags &= os.path.isfile(nml['field_files']['ne_name'])
            flags &= os.path.isfile(nml['field_files']['n0_name'])
            flags &= os.path.isfile(nml['field_files']['elec_name'])
            flags &= os.path.isfile(nml['field_files']['CX_name'])
        
        flags &= os.path.isfile(nml['scintillator']['triangle_file'])
        flags &= os.path.isfile(nml['deposition']['depos_file'])
            
    elif action == 'ihibpsim':
        flags =  os.path.isfile(nml['field_files']['bfield_name'])
        if nml['field_files']['efield_on']:
            flags &= os.path.isfile(nml['field_files']['efield_name'])
        if nml['tables']['beamattenuationmodule']:
            flags &= os.path.isfile(nml['field_files']['te_name'])
            flags &= os.path.isfile(nml['field_files']['ne_name'])
            flags &= os.path.isfile(nml['field_files']['n0_name'])
            flags &= os.path.isfile(nml['field_files']['elec_name'])
            flags &= os.path.isfile(nml['field_files']['CX_name'])
            flags &= os.path.isfile(nml['field_files']['prm_name'])
        
        flags &= os.path.isfile(nml['scint3d']['triangle_file'])
        flags &= os.path.isfile(nml['strikemapconf']['triangle_file'])
    
    elif action == 'shot_mapper':
        if nml['tables']['beamattenuationmodule']:
            flags &= os.path.isfile(nml['field_files']['elec_name'])
            flags &= os.path.isfile(nml['field_files']['CX_name'])
            flags &= os.path.isfile(nml['field_files']['prm_name'])
        
        flags &= os.path.isfile(nml['scint3d']['triangle_file'])
        flags &= os.path.isfile(nml['strikemapconf']['triangle_file'])
    return flags
    

def check_files_many(runID: str, action: str='shot_remap'):
    """
    Check if the needed files for the execution of the codes are properly
    stored in the corresponding folders.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param runID: identifier of the run.
    @param action: code to run. To be chosen among IHIBPSIM_ACTION_NAMES
    """
    
    # Cheking if the corresponding folder exists.
    path = os.path.join(paths.ihibp_res, runID)
    if not os.path.isdir(path):
        raise NotADirectoryError('The directory corresponding to the  '+\
                                 ' run ID = %s does not exist'%runID)
    
    nmls = [ii for ii in os.listdir(path) if ii.endswith('.cfg')]
    
    ret_flags = np.zeros((len(nmls), ), dtype=bool)
    for ii, nml in enumerate(nmls):
        ret_flags[ii] = check_files(nml, action=action)
    return ret_flags

