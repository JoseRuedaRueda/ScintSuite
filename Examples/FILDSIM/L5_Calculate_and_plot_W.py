"""
Calculate FILD instrument (or weight) function

Jose Rueda Rueda: jrrueda@us.es

Note: Done for version 0.7.3 of the suite

Please see L$ example of the SINPA code for an updated version.
"""
import Lib as ss
import numpy as np

# --- Settings:
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/Panos_strike_map.dat'
# Tomography parameters
s_opt = {
    'rmin': 1.2,
    'rmax': 8.0,
    'dr': 0.1,
    'pmin': 20.0,
    'pmax': 85.0,
    'dp': 1.0
}
p_opt = {
    'rmin': 1.2,
    'rmax': 8.0,
    'dr': 0.1,
    'pmin': 20.0,
    'pmax': 85.0,
    'dp': 1.0
}
diag_params = {
    'p_method': 'Gauss',
    'g_method': 'sGauss'
}
include_efficiency = True
B = 1.64
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function
# -----------------------------------------------------------------------------
# Calculate resolutions
smap = ss.mapping.StrikeMap('FILD', smap_file)
smap.load_strike_points()
smap.calculate_resolutions(diag_params=diag_params)
# --- create the grids
nr = int((s_opt['rmax'] - s_opt['rmin']) / s_opt['dr'])
nnp = int((s_opt['pmax'] - s_opt['pmin']) / s_opt['dp'])
sredges = s_opt['rmin'] - s_opt['dr']/2 + np.arange(nr+2) * s_opt['dr']
spedges = s_opt['pmin'] - s_opt['dp']/2 + np.arange(nnp+2) * s_opt['dp']

scint_grid = {'nr': nr + 1, 'np': nnp + 1,
              'r': 0.5 * (sredges[:-1] + sredges[1:]),
              'p': 0.5 * (spedges[:-1] + spedges[1:])}

nr = int((p_opt['rmax'] - p_opt['rmin']) / p_opt['dr'])
nnp = int((p_opt['pmax'] - p_opt['pmin']) / p_opt['dp'])
redges = p_opt['rmin'] - p_opt['dr']/2 + np.arange(nr+2) * p_opt['dr']
pedges = p_opt['pmin'] - p_opt['dp']/2 + np.arange(nnp+2) * p_opt['dp']
pin_grid = {'nr': nr + 1, 'np': nnp + 1,
            'r': 0.5 * (redges[:-1] + redges[1:]),
            'p': 0.5 * (pedges[:-1] + pedges[1:])}
if include_efficiency:
    eff = ss.scintcharact.ScintillatorEfficiency()
else:
    eff = None
W4D, W2D = ss.fildsim.build_weight_matrix(smap, scint_grid['r'],
                                          scint_grid['p'], pin_grid['r'],
                                          pin_grid['p'], eff,
                                          B=B)
ss.fildsim.plot_W(W4D, pin_grid['r'], pin_grid['p'], scint_grid['r'],
                  scint_grid['p'], pp0=55, pr0=5.0)
