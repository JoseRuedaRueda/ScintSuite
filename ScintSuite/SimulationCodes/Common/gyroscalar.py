import unyt
import numpy as np
import logging
logger = logging.getLogger(__name__)

__all__ = ['get_energy', 'get_gyroradius', 'get_velocity', 'get_beta']

def get_energy(gyroradius, B: float, A: float = 2.01410178, Z: float = 1.0,
               relativistic: bool = False):
    """
    Calculate the energy given a gyroradius, FILDSIM criteria

    Jose Rueda-Rueda: jruedaru@uci.edu
    
    :param  gyroradius: Larmor radius as taken from FILD strike map [in cm]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number (A=0 means electrons)
    :param  Z: Ion charge [in e units] (if A=0, Z=1, electron charge)
    :param  relativistic: Flag to use relativistic formula or not
    """
    if relativistic:
        return get_energy_relativistic(gyroradius, B, A, Z)
    else:
        return get_energy_classic(gyroradius, B, A, Z)


def get_energy_classic(gyroradius, B: float, A: float = 2.01410178, Z: float = 1.0):
    """
    Calculate the energy given a gyroradius, FILDSIM criteria

    jose Rueda: jrrueda@us.es

    :param  gyroradius: Larmor radius as taken from FILD strike map [in cm]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass, [in amu]
    :param  Z: Ion charge [in e units]

    :return E: the energy [in eV]
    """
    # Check if the gyroradius have units
    if not isinstance(gyroradius, unyt.unyt_array):
        logger.warning('The input gyroradius does not have units, assuming cm')
        gyroradius = gyroradius * unyt.cm
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = Z * unyt.electron_charge
    # E[ev] = (1/2 * (r[cm]/100*Z*B[T])**2 / m) * ec
    E = (0.5 * (gyroradius * Z * B)**2 / A) 
    return E.to('eV')

def get_energy_relativistic(gyroradius, B: float, A: float = 2.01410178, 
                            Z: float = 1.0):
    """
    Calculate the energy given a gyroradius, FILDSIM criteria
    Relativistic correction, implemented to include runaway electrons

    Alex Reyner: alereyvinn@alum.us.es .ft. Jose Rueda-Rueda: jruedaru@uci.edu

    :param  gyroradius: Larmor radius as taken from FILD strike map [in cm]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number
    :param  Z: Ion charge [in e units]

    :return E: the energy [in eV]
    """
    # Check if the gyroradius have units
    if not isinstance(gyroradius, unyt.unyt_array):
        logger.warning('The input gyroradius does not have units, assuming cm')
        gyroradius = gyroradius * unyt.cm
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = Z * unyt.electron_charge
    # Set the case for an electron
    if A == 0:
        Z = unyt.electron_charge
        m = unyt.electron_mass
    else:
        m = A
    gamma = np.sqrt((gyroradius * Z *B / m)**2 /unyt.speed_of_light**2 + 1) 

    E = (gamma-1) *(m *unyt.speed_of_light**2)

    return E.to('eV')

def get_gyroradius(E:float, B: float, A: float = 2.01410178, Z: float = 1.0, 
                   relativistic: bool = False):
    """
    Calculate the gyroradius given an energy, FILDSIM criteria

    Jose Rueda-Rueda: jruedaru@uci.edu
    
    :param  energy: Energy [eV]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number (A=0 means electrons)
    :param  Z: Ion charge [in e units] (if A=0, Z=1, electron charge)
    :param  relativistic: Flag to use relativistic formula or not
    """
    if relativistic:
        return get_gyroradius_relativistic(E, B, A, Z)
    else:
        return get_gyroradius_classic(E, B, A, Z)

def get_gyroradius_classic(E, B: float, A: float = 2.01410178, Z: float = 1.0):
    """
    Calculate the gyroradius given an energy, FILDSIM criteria

    jose Rueda: jrrueda@us.es

    :param  energy: Energy [eV]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number
    :param  Z: Ion charge [in e units]

    :return r: Larmor radius as taken from FILD strike map [in cm]
    """
    # r[cm] = sqrt(2*(E[eV]/ec)*m) / (Z*B[T]) * 100
        # Check if the gyroradius have units
    if not isinstance(E, unyt.unyt_array):
        logger.warning('The input energy does not have units, assuming eV')
        E = E * unyt.eV
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = -1.0 *Z * unyt.electron_charge
    r = np.sqrt(2.0 * E * A) / (Z*B)
    return r.to('cm')

def get_gyroradius_relativistic(E, B: float, A: float = 2.01410178, 
                                Z: float = 1.0):
    """
    Calculate the gyroradius given an energy, FILDSIM criteria
    Relativistic correction, implemented to include runaway electrons

    Alex Reyner: areyner@us.es .ft. Jose Rueda-Rueda: jruedaru@uci.edu

    :param  energy: Energy [eV]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number. 0 means electrons
    :param  Z: Ion charge [in e units]

    :return r: Larmor radius as taken from FILD strike map [in cm]
    """
    if not isinstance(E, unyt.unyt_array):
        logger.warning('The input energy does not have units, assuming eV')
        E = E * unyt.eV
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = -1.0 *Z * unyt.electron_charge
    
    if A == 0:
        m = unyt.electron_mass
        Z = unyt.electron_charge
    else:
        m = A
    gamma = 1 + (E/(m*unyt.speed_of_light**2))
    beta = np.sqrt(1-1/gamma**2)
    v = unyt.speed_of_light * beta
    r = (gamma*m*v) / (Z*B)

    return r.to('cm')

def get_beta(E, B: float, A: float = 2.01410178, Z: float = 1.0):
    """
    Calculate the relativistic factor

    Alex Reyner: areyner@us.es .ft. Jose Rueda-Rueda:jruedaru@uci.edu

    :param  energy: Energy [eV]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number. 0 means electrons
    :param  Z: Ion charge [in e units]

    :return beta: v/c
    """
    if not isinstance(E, unyt.unyt_array):
        logger.warning('The input energy does not have units, assuming eV')
        E = E * unyt.eV
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = -1.0 *Z * unyt.electron_charge
    
    if A == 0:
        m = unyt.electron_mass
        Z = unyt.electron_charge
    else:
        m = A
    gamma = 1 + (E/(m*unyt.speed_of_light**2))
    beta = np.sqrt(1-1/gamma**2)

    return beta

def get_velocity(gyroradius, B: float = 1.9, A: float = 2.01410178, Z: float = 1.0):
    """
    Calculate the velocity given a gyroradius, FILDSIM criteria
    Relativistic correction, implemented to include runaway electrons

    Alex Reyner: alereyvinn@alum.us.es .ft. Jose Rueda-Rueda: jruedaru@uci.edu

    :param  gyroradius: Larmor radius as taken from FILD strike map [in cm]
    :param  energy: Energy [eV]
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number
    :param  Z: Ion charge [in e units]

    :return E: the energy [in eV]
    """
    if not isinstance(gyroradius, unyt.unyt_array):
        logger.warning('The input gyroradius does not have units, assuming cm')
        gyroradius = gyroradius * unyt.cm
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = -1.0 *Z * unyt.electron_charge

    E = get_energy_relativistic(gyroradius,B,A,Z)
    beta = get_beta(E,B,A,Z)
    v = beta * unyt.speed_of_light

    return v.to('m/s')

def get_gyrofrequency(B: float = 1.9, A: float = 2.01410178, Z: float = 1.0):
    """
    Calculate the gyrofrequency of a particle, given a magnetic field and particle properties
    
    Jose Rueda-Rueda: jruedaru@uci.edu
    
    :param  B: Magnetic field, [in T]
    :param  A: Ion mass number (A=0 means electrons)
    :param  Z: Ion charge [in e units] (if A=0, Z=1, electron charge)
    :return f: gyrofrequency 
    """
    if not isinstance(B, unyt.unyt_array):
        logger.warning('The input magnetic field does not have units, assuming T')
        B = B * unyt.T
    if not isinstance(A, unyt.unyt_array):
        logger.warning('The input mass number does not have units, assuming amu')
        A = A * unyt.amu
    if not isinstance(Z, unyt.unyt_array):
        logger.warning('The input charge number does not have units, assuming e')
        Z = Z * unyt.elementary_charge
    
    if A == 0:
        m = unyt.electron_mass
        Z = unyt.elementary_charge
    else:
        m = A

    f = (Z*B) / m

    return f.to('Hz')