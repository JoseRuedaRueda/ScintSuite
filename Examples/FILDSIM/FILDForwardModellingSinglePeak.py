"""
Calculate FILD synthetic signal of a single energy peak

Jose Rueda: jrrueda@us.es

Gaussian input distribution will be used

done for version 0.4.2
"""
import Lib as ss
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
# As the strike points are needed and they are not included in the remap
# database, for the tomography one should manually select (for now) the strike
# map
smap_file = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_map.dat'
smap_points = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_points.dat'

# General options:
r0 = 2.0        # centroid of the distribution in gyroradius
sr0 = 0.1       # sigma of the distribution in gyroradius
p0 = 55.0       # centroid of the distribution in pitch
sp0 = 3        # sigma of the distribution in pitch
efficiency = True
diag_params = {
    'g_method': 'sGauss',
    'p_method': 'Gauss'
}
# -----------------------------------------------------------------------------
# --- input distribution
# -----------------------------------------------------------------------------
# You can set as inputs here the magnetic field, the mass and charge, for
# the calculation of the relation gyroradius <-> energy
input = ss.fildsim.gaussian_input_distribution(r0, sr0, p0, sp0)

# -----------------------------------------------------------------------------
# --- Efficiecncy
# -----------------------------------------------------------------------------
if efficiency:
    # The default is D over TgGreeA of 9 mu m
    eff = ss.scintcharact.ScintillatorEfficiency()
else:
    eff = None
# -----------------------------------------------------------------------------
# --- synthetic signal
# -----------------------------------------------------------------------------
g_grid, p_grid, signal = \
    ss.fildsim.synthetic_signal(input, smap_file, spoints=smap_points,
                                diag_params=diag_params, efficiency=eff)
