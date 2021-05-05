"""Calculate and plot synthetic signals with FILDSIM"""
import os
import numpy as np
import matplotlib.pyplot as plt
import LibMap as ssmapping
import FILDSIMexecution as ssfildsimA
from LibMachine import machine
import LibPlotting as ssplt
import matplotlib.pyplot as plt
if machine == 'AUG':
    import LibDataAUG as ssdat
try:
    import lmfit
except ModuleNotFoundError:
    print('lmfit not found, not possible foward modelling')


# -----------------------------------------------------------------------------
# --- Inputs distributions
# -----------------------------------------------------------------------------
def gaussian_input_distribution(r0, sr0, p0, sp0, B=1.8, A=2.0, Z=1):
    """
    Prepare an 2D Gaussian input distribution

    Jose Rueda Rueda: jrrueda@us.es

    @param r0: centroid of the Gaussian in gyroradius
    @param sr0: sigma of the gaussian in gyroradius
    @param p0: centroid of the gaussian in pitch
    @param sp0: sigma of the gaussian in pitch
    @param B: Magnetic field (to translate from r to Energy)
    @param A: Mass, in amu, (to translate from r to Energy)
    @param Z: Charge in e units (to translate from r to Energy)

    @return: distribution ready to be used by the foward model routine
    """
    gaussian = lmfit.models.GaussianModel().func

    ggrid = np.linspace(r0 - 3 * sr0, r0 + 3 * sr0)
    pgrid = np.linspace(p0 - 3 * sp0, p0 + 3 * sp0)

    G, P = np.meshgrid(ggrid, pgrid)
    weights = gaussian(G.flatten(), center=r0, sigma=sr0) \
        * gaussian(P.flatten(), center=p0, sigma=sp0)
    distro = {
        'pitch': P.flatten(),
        'gyroradius': G.flatten(),
        'weight': weights.flatten(),
        'histograms': {
            'gp': {
                'gyroradius': ggrid,
                'pitch': pgrid,
                'hist': np.reshape(weights, G.shape).T
            }
        }
    }
    distro['energy'] = ssfildsimA.get_energy(distro['gyroradius'],
                                             B, A, Z)
    distro['n'] = len(distro['weight'])
    return distro


def read_ASCOT_distribution(file, version: int = 4, IpBt_sign=ssdat.IB_sign):
    """
    Read a distribution coming from ASCOT

    Jose Rueda: jrrueda@us.es

    @param file: full path to the file
    @param version: ASCOT version, default 4
    """
    out = {}
    if version == 4:
        print('Reading file: ', file)
        # --- Read header:
        header = []
        with open(file) as f:
            for j in range(30):
                header.append(f.readline())
        # extract names:
        names = []
        help = []
        units = []
        for i in range(10, 29):
            nam, h = header[i].split('-')
            names.append(nam.strip())
            help.append(h.strip())
            dummy, dummy2 = h.split('(')
            uni, dummy = dummy2.split(')')
            units.append(uni.strip())
        out['header'] = header
        out['help'] = help
        out['units'] = units
        # --- Read data
        # ASCOT4 gives us 30 line of comments:
        data = np.loadtxt(file, skiprows=30)
        for i in range(len(names)):
            out[names[i]] = data[:, i]
        out['n'] = len(data[:, 0])
        # --- Try to calculate the total field:
        if 'B' not in out.keys():
            try:
                out['B'] = np.sqrt(out['Bphi']**2 + out['Bz']**2
                                   + out['BR']**2)
            except KeyError:
                print('Not possible to calculate B')

        # --- Try to calculate the pitch:
        if 'pitch' not in out.keys():
            try:
                vmod = np.sqrt(out['vR']**2 + out['vphi']**2 + out['vz']**2)
                bmod = np.sqrt(out['BR']**2 + out['Bphi']**2 + out['Bz']**2)
                pitch = IpBt_sign * (out['vR'] * out['BR']
                                     + out['vphi'] * out['Bphi']
                                     + out['vz'] * out['Bz']) / vmod / bmod
                out['pitch'] = pitch
            except KeyError:
                print('Not possible to calculate pitch')
        if 'gyroradius' not in out.keys():
            try:
                r = ssfildsimA.get_gyroradius(out['energy'], out['B'],
                                              out['Anum'], out['Znum'])
                out['gyroradius'] = r
            except KeyError:
                print('Not possible to calculate gyroradius')
        return out


def distribution_tomography_frame(g_grid, p_grid, coefficients,
                                  exclude_neg: bool = True):
    """
    Prepare distribution for foward modelling from tomographic inversions:

    Jose Rueda Rueda: jrrueda@us.es

    @param g_grid: array with gyroradius values
    @param p_grid: array with pitch values
    @param coefficients: 2D array with the inversion results, [ngyr, npitch]
    @param exclude_neg: if True, negative values will be excluded
    """
    # prepare the grid:
    gyr, pitch = np.meshgrid(g_grid, p_grid)
    dummy = coefficients.T
    if exclude_neg:
        flags = dummy > 0
        gyr = gyr[flags]
        pitch = pitch[flags]
        dummy = dummy[flags]
    distro = {
        'pitch': pitch.flatten(),
        'gyroradius': gyr.flatten(),
        'weight': coefficients.flatten(),
    }
    distro['n'] = len(distro['weight'])
    return distro


# -----------------------------------------------------------------------------
# --- Synthetic signals
# -----------------------------------------------------------------------------
def synthetic_signal(distro, smap, spoints=None, diag_params: dict = {},
                     gmin=1.5, gmax=10.0, dg=0.1, pmin=20.0, pmax=90.0,
                     dp=1.0, efficiency=None):
    """
    Generate FILD synthetic signal

    Jose Rueda: jrrueda@us.es

    Based on the matlab implementation written by Joaquin Galdón

    @param distro: distribution, created by one of the routines of this library
    , example (read_ASCOT_distribution(), or distribution_tomography_frame())
    @param smap: Strike map object or path pointing to the strike map file
    @param spoints: path pointing to the strike point file. Not needed if smap
    is a strike map object with the resolutions already calculated
    @param diag_params: Parametes for the resolution calculation, useless if
    the input strike map has the resolutions already calcualted See
    StrikeMap.calculate_resolutions() for the whole list of options
    @param gmin: Minimum gyroradius to consider in the synthetic signal
    @param gmax: Maximum gyroradius to consider in the synthetic signal
    @param dg: space in the gyroradius
    @param pmin: Minimum pitch to consider in the synthetic signal
    @param pmax: Maximum pitch to consider in the synthetic signal
    @param dp: space in pitch
    @param efficiency: ScintillatorEfficiency() object. If None, efficiency
    will not be included

    @return g_grid: gyroradius where the synthetic signal was evaluated
    @return p_grid: pitch where the synthetic signal was evaluated
    @return signal: Synthetic signal[ngyr, npitch]
    """
    # Initialise the diag_params:
    diag_parameters = {
        'g_method': 'sGauss',
        'p_method': 'Gauss'
    }
    diag_parameters.update(diag_params)
    # Check / load the strike map
    if isinstance(smap, str):  # if it is string, load the file.
        if spoints is None:
            raise Exception('StrikePoints file needed!!')
        print('Reading strike map: ', smap)
        smap = ssmapping.StrikeMap(file=smap)
        smap.load_strike_points(spoints)
    if smap.resolution is None:
        smap.calculate_resolutions(diag_params=diag_parameters)

    # --- Prepare the models for the signal calculation:
    # Just for consitency, use the same one we use for the resolution
    # calculation
    if smap.resolution['pitch_model'] == 'Gauss':
        pitch_func = lmfit.models.GaussianModel().func
    elif smap.resolution['pitch_model'] == 'sGauss':
        pitch_func = lmfit.models.SkewedGaussianModel().func

    if smap.resolution['gyroradius_model'] == 'Gauss':
        g_func = lmfit.models.GaussianModel().func
    elif smap.resolution['gyroradius_model'] == 'sGauss':
        g_func = lmfit.models.SkewedGaussianModel().func
    # --- Calculate the signal:
    # Prepare the grid
    ng = int((gmax-gmin)/dg)
    npitch = int((pmax-pmin)/dp)
    g_array = gmin + np.arange(ng+1) * dg
    p_array = pmin + np.arange(npitch+1) * dp
    g_grid, p_grid = np.meshgrid(g_array, p_array)
    # Prepare the parameters we will interpolate:
    parameters_to_consider = {
        'Gauss': ['sigma', 'center'],
        'sGauss': ['sigma', 'gamma', 'center']
    }
    signal = np.zeros(g_grid.size)
    # Make the comparison if efficiency is None or not, to avoid doing it
    # inside the loop:
    if efficiency is not None:
        eff = True
    else:
        eff = False

    for i in range(distro['n']):
        # Interpolate sigmas, gammas and collimator_factor
        g_parameters = {}
        for k in parameters_to_consider[smap.resolution['gyroradius_model']]:
            g_parameters[k] = \
                smap.interpolators['gyroradius'][k]\
                (distro['gyroradius'][i], distro['pitch'][i])
        p_parameters = {}
        for k in parameters_to_consider[smap.resolution['pitch_model']]:
            p_parameters[k] = \
                smap.interpolators['pitch'][k]\
                (distro['gyroradius'][i], distro['pitch'][i])

        col_factor = smap.interpolators['collimator_factor']\
            (distro['gyroradius'][i], distro['pitch'][i]) / 100.0

        if eff:
            signal += col_factor * g_func(g_grid.flatten(), **g_parameters) \
                * pitch_func(p_grid.flatten(), **p_parameters)\
                * distro['weight'][i]\
                * efficiency.interpolator(distro['energy'][i])
        else:
            signal += col_factor * g_func(g_grid.flatten(), **g_parameters) \
                * pitch_func(p_grid.flatten(), **p_parameters)\
                * distro['weight'][i]
    signal = np.reshape(signal, g_grid.shape)

    return g_array, p_array, signal.T


def plot_synthetic_signal(r, p, signal, cmap=None, ax=None, ax_params={}):
    """
    Plot the synthetic signal

    Jose Rueda: jrrueda@us.es

    @param r: array with gyroradius values
    @param p: array with pitch values
    @param signal: matrix with the signal [np, nr]
    @param cmap: color map to use, if none: ssplt.Gamma_II()
    @param ax: axes where to plot, if none, a new figure will be created
    @param ax_params: only used if the axis was created here, parameters for
    the axis_beauty function

    @return ax: axes where the figure was drawn
    """
    # Initialise the axis options:
    ax_options = {
        'xlabel': 'Pitch',
        'ylabel': 'Gyroradius [cm]',
        'fontsize': 14,
    }
    ax_options.update(ax_params)

    # Prepare the color map:
    if cmap is None:
        cmap = ssplt.Gamma_II()

    # Open the axis:
    if ax is None:
        fig, ax = plt.subplots()
        created = True
    else:
        created = False
    # plot:
    ax.contourf(r, p, signal.T, cmap=cmap)
    if created:
        ax = ssplt.axis_beauty(ax, ax_options)
    return ax


# -----------------------------------------------------------------------------
# --- Weight function
# -----------------------------------------------------------------------------
def build_weight_matrix(smap, rscint, pscint, rpin, ppin,
                        efficiency=None, spoints=None, diag_params: dict = {},
                        B = 1.8, A = 2.0, Z = 1):
    """
    Build FILD weight function

    Jose Rueda Rueda: jrrueda@us.es

    Introduced in version 0.4.2

    @param smap: Strike map object or path pointing to the strike map file
    @param spoints: path pointing to the strike point file. Not needed if smap
    is a strike map object with the resolutions already calculated
    @param diag_params: Parametes for the resolution calculation, useless if
    the input strike map has the resolutions already calcualted See
    StrikeMap.calculate_resolutions() for the whole list of options
    @param sgmin: Minimum gyroradius to consider in the scintillator
    @param sgmax: Maximum gyroradius to consider in the scintillator
    @param sdg: space in the gyroradius scintillator
    @param spmin: Minimum pitch to consider in the scintillator
    @param spmax: Maximum pitch to consider in the scintillator
    @param sdp: space in pitch scintillator
    @param pgmin: Minimum gyroradius to consider in the pinhole
    @param pgmax: Maximum gyroradius to consider in the pinhole
    @param pdg: space in the gyroradius pinhole
    @param ppmin: Minimum pitch to consider in the pinhole
    @param ppmax: Maximum pitch to consider in the pinhole
    @param pdp: space in pitch pinhole
    @param efficiency: ScintillatorEfficiency() object. If None, efficiency
    will not be included
    """
    # --- Initialise the diag_params:
    diag_parameters = {
        'g_method': 'sGauss',
        'p_method': 'Gauss'
    }
    diag_parameters.update(diag_params)
    # --- Check the StrikeMap
    if isinstance(smap, str):  # if it is string, load the file.
        if spoints is None:
            raise Exception('StrikePoints file needed!!')
        print('Reading strike map: ', smap)
        smap = ssmapping.StrikeMap(file=smap)
        smap.load_strike_points(spoints)
    if smap.resolution is None:
        smap.calculate_resolutions(diag_params=diag_parameters)

    print('Calculating FILD weight matrix')
    # Prepare the grid:
    nr_scint = len(rscint)
    np_scint = len(pscint)

    dr_scint = abs(rscint[1] - rscint[0])
    dp_scint = abs(pscint[1] - pscint[0])

    # Pinhole grid
    nr_pin = len(rpin)
    np_pin = len(ppin)

    # --- Prepare model and efficiecncy
    # Prepare the model based on the resolution calculation:
    # Just for consitency, use the same one we use for the resolution
    # calculation
    if smap.resolution['pitch_model'] == 'Gauss':
        pitch_func = lmfit.models.GaussianModel().func
    elif smap.resolution['pitch_model'] == 'sGauss':
        pitch_func = lmfit.models.SkewedGaussianModel().func

    if smap.resolution['gyroradius_model'] == 'Gauss':
        g_func = lmfit.models.GaussianModel().func
    elif smap.resolution['gyroradius_model'] == 'sGauss':
        g_func = lmfit.models.SkewedGaussianModel().func
    # prepare the grid
    Rscint, Pscint = np.meshgrid(rscint, pscint)
    # Prepare the parameters we will interpolate:
    parameters_to_consider = {
        'Gauss': ['sigma', 'center'],
        'sGauss': ['sigma', 'gamma', 'center']
    }
    # Make the comparison if efficiency is None or not, to avoid doing it
    # inside the loop:
    if efficiency is not None:
        eff = True
        energy = ssfildsimA.get_energy(rpin, B, A, Z)
        eff = efficiency.interpolator(energy)
        print('considering scintillator efficiency in W')
    else:
        eff = np.ones(rpin.size)
    # Build the weight matrix. We will use brute force, I am sure that there is
    # a tensor product implemented in python which does the job in a more
    # efficient way, bot for the moment, I will leave exactly as in the
    # original IDL routine
    res_matrix = np.zeros((nr_scint, np_scint, nr_pin, np_pin))

    print('Creating matrix')
    for kk in range(nr_pin):
        for ll in range(np_pin):
            # Interpolate sigmas, gammas and collimator_factor
            g_parameters = {}
            for k in parameters_to_consider[smap.resolution['gyroradius_model']]:
                g_parameters[k] = \
                    smap.interpolators['gyroradius'][k](rpin[kk], ppin[ll])

            p_parameters = {}
            for k in parameters_to_consider[smap.resolution['pitch_model']]:
                p_parameters[k] = \
                    smap.interpolators['pitch'][k](rpin[kk], ppin[ll])

            col_factor = smap.interpolators['collimator_factor']\
                (rpin[kk], ppin[ll]) / 100.0
            if col_factor > 0.0:
                # Calculate the contribution:
                dummy = col_factor * g_func(Rscint.flatten(), **g_parameters) \
                    * pitch_func(Pscint.flatten(), **p_parameters)\
                    * eff[kk] * dr_scint * dp_scint
                res_matrix[:, :, kk, ll] = np.reshape(dummy, Rscint.shape).T
            else:
                res_matrix[:, :, kk, ll] = 0.0
    res_matrix[np.isnan(res_matrix)] = 0.0
    return res_matrix


def plot_W(W4D, pr, pp, sr, sp, pp0=None, pr0=None, sp0=None, sr0=None,
           cmap=None, nlev=20):
    """
    Plot the weight function

    Jose Rueda Rueda: jrrueda@us.es

    @todo: add titles and print the used point

    @param W4D: 4-D weight function
    @param pr: array of gyroradius at the pinhole used to calculate W
    @param pp: array of pitches at the pinhole used to calculate W
    @param sr: array of gyroradius at the scintillator used to calculate W
    @param sp: array of pitches at the scintillator used to calculate W
    @param pp0: precise radius wanted at the pinhole to plot the scintillator W
    @param pr0: precise pitch wanted at the pinhole to plot the scintillator W
    @param sp0: precise radius wanted at the pinhole to plot the scintillator W
    @param sr0: precise pitch wanted at the pinhole to plot the scintillator W
    """
    # --- Color map
    if cmap is None:
        ccmap = ssplt.Gamma_II()
    # --- Potting of the scintillator weight
    # We will select a point of the pinhole and see how it seen in the
    # scintillator
    if (pp0 is not None) and (pr0 is not None):
        ip = np.argmin(abs(pp - pp0))
        ir = np.argmin(abs(pr - pr0))
        W = W4D[:, :, ir, ip]
        fig, ax = plt.subplots()
        a = ax.contourf(sp, sr, W, nlev, cmap=ccmap)
        plt.colorbar(a, ax=ax)
    if (sp0 is not None) and (sr0 is not None):
        ip = np.argmin(abs(pp - sp0))
        ir = np.argmin(abs(pr - sr0))
        W = W4D[ir, ip, :, :]
        fig, ax = plt.subplots()
        a = ax.contourf(pp, pr, W, nlev, cmap=ccmap)
        plt.colorbar(a, ax=ax)
