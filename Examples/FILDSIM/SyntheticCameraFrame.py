"""
Generate a synthetic camera frame

Jose Rueda: jrrueda@us.es

Lines which end with a '# -#' should be changed in order to run the example

Created for version 0.4.13

Note: The plotted strike map and scintillator in the final picture is not
distorted!!!! Distortion is just applied to the camera frames
"""
import Lib as ss

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
# As the strike points are needed and they are not included in the remap
# database, for the tomography one should manually select (for now) the strike
# map
smap_file = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_map.dat'                          # -#
smap_points = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_points.dat'                       # -#
scintillator = \
    '/afs/ipp/home/r/ruejo/FILDSIM/geometry/aug_fild1_scint_v6.pl'         # -#
# General options:
r0 = 3.31        # centroid of the distribution in gyroradius
sr0 = 0.05       # sigma of the distribution in gyroradius
p0 = 64.0        # centroid of the distribution in pitch
sp0 = 3          # sigma of the distribution in pitch

diag_params = {  # for the model to calculate strikes in the scintillator
    'g_method': 'sGauss',
    'p_method': 'Gauss'
}
camera_model = 'PhantomV2512'
t_influx = 10**13       # ions per second at the pinhole
exp_time = 1e-4         # exposure time, in s
# -- Optics parameters:
optics = {
    'beta': 1.0/11.0,   # magnification factor
    'T': 0.95,          # transmission factor
    'Omega': 1e-4       # covered solid angle
}
distortion_options = {
    'model': 'WandImage',  # Distortion implementation, only this is done
    'parameters': {
        'method': 'barrel',  # type of distortion
        'arguments': (0.2, 0.1, 0.1, 0.6)  # A,B,C,D
    },
}
# -- Noise options:
noise_options = {
    'dark_readout': {   # Apply dark current with the parameters of the camera
        'apply': True   # file
    },
    'camera_neutrons': {
        'percent': 0.001,  # percent (normlalised to 1) of neutron affected
        'vmin': 0.7,       # pixels
        'vmax': 0.9,
    },
    'broken': {
        'percent': 0.001  # percent (normlalised to 1) of broken pixels
    },
    'photon': {
        'multiplier': 1.0  # 'Fano' factor of the photons noise
    }
}
# -----------------------------------------------------------------------------
# --- input distribution
# -----------------------------------------------------------------------------
# You can set as inputs here the magnetic field, the mass and charge, for
# the calculation of the relation gyroradius <-> energy
input = ss.fildsim.gaussian_input_distribution(r0, sr0, p0, sp0, F=t_influx)
# -----------------------------------------------------------------------------
# --- Efficiecncy
# -----------------------------------------------------------------------------
# The default is D over TgGreeA of 9 mu m
eff = ss.scintcharact.ScintillatorEfficiency()
# -----------------------------------------------------------------------------
# --- synthetic signal
# -----------------------------------------------------------------------------
synthetic_frame = \
    ss.fildsim.synthetic_signal(input, eff, optics,
                                smap_file, scintillator, camera_model,
                                exp_time=exp_time,
                                spoints=smap_points,
                                diag_parameters=diag_params,
                                distortion_params=distortion_options,
                                noise_params=noise_options,
                                plot=True)
