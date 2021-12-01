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

#-----------------------------------------------------------------------------
# Routines to make namelists to run iHIBPsim.
#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
# Routines to check the namelist consistency.
#-----------------------------------------------------------------------------
def check_namelist(params:dict, codename: str='ihibpsim'):
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
        return __check_tracker_namelist(params)
    elif codename == 'ihibpsim':
        return __check_ihibpsim_namelist(params)
    elif codename == 'shot_mapper':
        return __check_shotmapper_namelist(params)
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
    return __check_ihipbsim_geometry(params['geom'])

def __check_shotmapper_namelist(params: dict):
    pass


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