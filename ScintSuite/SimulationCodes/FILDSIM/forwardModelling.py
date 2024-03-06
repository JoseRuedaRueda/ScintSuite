"""Calculate and plot synthetic signals with FILDSIM."""
import numpy as np
import matplotlib.pyplot as plt
import ScintSuite._Mapping as ssmapping
import ScintSuite.SimulationCodes.FILDSIM.execution as ssfildsimA
import ScintSuite._Plotting as ssplt
import ScintSuite.LibData as ssdat
import ScintSuite._Noise as ssnoise
import ScintSuite._IO as ssio
import ScintSuite._Optics as ssoptics
from tqdm import tqdm            # For waitbars
try:
    import lmfit
except ModuleNotFoundError:
    print('lmfit not found, not possible foward modelling')


# -----------------------------------------------------------------------------
# --- Inputs distributions
# -----------------------------------------------------------------------------
def gaussian_input_distribution(r0, sr0, p0, sp0, B=1.8, A=2.0, Z=1, F=1e6,
                                n=50):
    """
    Prepare an 2D Gaussian input distribution

    Jose Rueda Rueda: jrrueda@us.es

    :param  r0: centroid of the Gaussian in gyroradius
    :param  sr0: sigma of the gaussian in gyroradius
    :param  p0: centroid of the gaussian in pitch
    :param  sp0: sigma of the gaussian in pitch
    :param  B: Magnetic field (to translate from r to Energy)
    :param  A: Mass, in amu, (to translate from r to Energy)
    :param  Z: Charge in e units (to translate from r to Energy)
    :param  F: Sum of the weight of the generated markers

    :return: distribution ready to be used by the foward model routine
    """
    gaussian = lmfit.models.GaussianModel().func
    # 50 points in 6 sigmas sounds a finner enought grid
    ggrid = np.linspace(r0 - 3 * sr0, r0 + 3 * sr0, num=n)
    pgrid = np.linspace(p0 - 3 * sp0, p0 + 3 * sp0, num=n)

    G, P = np.meshgrid(ggrid, pgrid)
    weights = gaussian(G.flatten(), center=r0, sigma=sr0) \
        * gaussian(P.flatten(), center=p0, sigma=sp0)
    distro = {
        'pitch': P.flatten(),
        'gyroradius': G.flatten(),
        'weight': weights.flatten() * F / np.sum(weights),
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


def read_ASCOT_distribution(file, version: int = 4, IpBt_sign=-1.0, B=None):
    """
    Read a distribution coming from ASCOT

    Jose Rueda: jrrueda@us.es

    :param  file: full path to the file
    :param  version: ASCOT version, default 4
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
        try:
            for i in range(10, 29):
                nam, h = header[i].split('-')
                names.append(nam.strip())
                help.append(h.strip())
                dummy, dummy2 = h.split('(')
                uni, dummy = dummy2.split(')')
                units.append(uni.strip())
        except ValueError:
            # We havve a dummy file with no header
            names = ['R', 'phi', 'z', 'energy', 'pitch', 
                     'Anum', 'Znum', 'weight', 'time']
            header = 'Hardcored format file'
            units = ['m', 'm', 'deg', 'eV', '', 'proton', 'e', 'ions/MC', 's']
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
                print('Not possible to calculate B, assuming from inputs')
                out['B'] = B*np.ones(out['n'])

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
    Prepare distribution for foward modelling from tomographic inversions

    Note: this function is dumm, the foward modelled signal from a tomography
    is just WF, why this function was created??

    Jose Rueda Rueda: jrrueda@us.es

    :param  g_grid: array with gyroradius values
    :param  p_grid: array with pitch values
    :param  coefficients: 2D array with the inversion results, [ngyr, npitch]
    :param  exclude_neg: if True, negative values will be excluded
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
def synthetic_signal_remap(distro, smap, spoints=None, diag_params: dict = {},
                           rmin=None, rmax=None, dr=0.1, pmin=None, pmax=None,
                           dp=1.0, efficiency=None):
    """
    Generate FILD synthetic signal

    Jose Rueda: jrrueda@us.es

    Based on the matlab implementation written by Joaquin Galdón

    :param  distro: distribution, created by one of the routines of this library
    , example (read_ASCOT_distribution(), or distribution_tomography_frame())
    :param  smap: Strike map object or path pointing to the strike map file
    :param  spoints: path pointing to the strike point file. Not needed if smap
    is a strike map object with the resolutions already calculated
    :param  diag_params: Parametes for the resolution calculation, useless if
    the input strike map has the resolutions already calcualted See
    StrikeMap.calculate_resolutions() for the whole list of options
    :param  gmin: Minimum gyroradius to consider in the synthetic signal
    :param  gmax: Maximum gyroradius to consider in the synthetic signal
    :param  dg: space in the gyroradius
    :param  pmin: Minimum pitch to consider in the synthetic signal
    :param  pmax: Maximum pitch to consider in the synthetic signal
    :param  dp: space in pitch
    :param  efficiency: ScintillatorEfficiency() object. If None, efficiency
    will not be included

    :return output. dictionary contaiong:
            'gyroradius': Array of gyroradius where the signal is evaluated
            'pitch': Array of pitches where the signal is evaluated
            'dgyr': spacing of the gyroradius array
            'dp': spaing of the signal array
            'signal': signal matrix [ngyr, npitch], the units will be the ones
            of the input, divided by dgyr and dpitch. And, if efficiency is
            included, multiplied by photons/ions
    
    TODO: Units of energy for the fficiecny calculation now are assumed as keV
    """
    # Initialise the diag_params:
    diag_parameters = {
        'g_method': 'sGauss',
        'p_method': 'Gauss'
    }
    diag_parameters.update(diag_params)
    # Check / load the strike map
    if isinstance(smap, str):  # if it is string, load the file.
        print('Reading strike map: ', smap)
        smap = ssmapping.StrikeMap(file=smap)
        if spoints is None:
            smap.load_strike_points()
        else:
            smap.load_strike_points(spoints)
    if smap._resolutions is None:
        smap.calculate_phase_space_resolution(diag_params=diag_parameters)

    # --- Prepare the models for the signal calculation:
    # Just for consitency, use the same one we use for the resolution
    # calculation
    if smap._resolutions['model_pitch'] == 'Gauss':
        pitch_func = lmfit.models.GaussianModel().func
    elif smap._resolutions['model_pitch'] == 'sGauss':
        pitch_func = lmfit.models.SkewedGaussianModel().func

    if smap._resolutions['model_gyroradius'] == 'Gauss':
        g_func = lmfit.models.GaussianModel().func
    elif smap._resolutions['model_gyroradius'] == 'sGauss':
        g_func = lmfit.models.SkewedGaussianModel().func
    # --- Calculate the signal:
    # Make the comparison if efficiency is None or not, to avoid doing it
    # inside the loop:
    if efficiency is not None:
        eff = True
    else:
        eff = False
    # Remove all the values outside the Smap, to avoid NaN:
    gmax = smap.MC_variables[1].data.max()
    gmin = smap.MC_variables[1].data.min()
    ppmax = smap.MC_variables[0].data.max()
    ppmin = smap.MC_variables[0].data.min()
    flags = (distro['gyroradius'] > gmin) * (distro['gyroradius'] < gmax) \
        * (distro['pitch'] > ppmin) * (distro['pitch'] < ppmax)
    flags = flags.astype(bool)
    distro_gyr = distro['gyroradius'][flags]
    distro_pitch = distro['pitch'][flags]
    distro_energy = distro['energy'][flags]
    distro_w = distro['weight'][flags]
    lost_weight_outside = np.sum(distro['weight'][~flags])
    lost_weight_inside = 0
    # Prepare the grid
    if rmax is None:
        rmax = gmax
    if rmin is None:
        rmin = gmin
    if pmin is None:
        pmin = ppmin
    if pmax is None:
        pmax = ppmax
    ng = int((rmax-rmin)/dr)
    npitch = int((pmax-pmin)/dp)
    g_array = rmin + np.arange(ng+1) * dr
    p_array = pmin + np.arange(npitch+1) * dp
    g_grid, p_grid = np.meshgrid(g_array, p_array)
    # Prepare the parameters we will interpolate:
    parameters_to_consider = {
        'Gauss': ['sigma', 'center'],
        'sGauss': ['sigma', 'gamma', 'center']
    }
    signal = np.zeros(g_grid.size)
    if np.sum(flags) < distro['n']:
        print('Some markers were outside the strike map!')
        print('We neglected ', str(distro['n'] - np.sum(flags)), ' markers')
    for i in range(len(distro_gyr)):
        # Interpolate sigmas, gammas and collimator_factor
        g_parameters = {}
        for k in parameters_to_consider[smap._resolutions['model_gyroradius']]:
            g_parameters[k] = \
                smap._interpolators_instrument_function['gyroradius'][k](
                    distro_pitch[i], distro_gyr[i])
        p_parameters = {}
        for k in parameters_to_consider[smap._resolutions['model_pitch']]:
            p_parameters[k] = \
                smap._interpolators_instrument_function['pitch'][k](
                    distro_pitch[i], distro_gyr[i])

        col_factor = smap._interpolators_instrument_function['collimator_factor'](
            distro_pitch[i], distro_gyr[i]) / 100.0

        if eff:
            signal += col_factor * g_func(g_grid.flatten(), **g_parameters) \
                * pitch_func(p_grid.flatten(), **p_parameters)\
                * distro_w[i] * efficiency(distro_energy[i]/1000.0)
        else:
            dummy = col_factor * g_func(g_grid.flatten(), **g_parameters) \
                    * pitch_func(p_grid.flatten(), **p_parameters)\
                    * distro_w[i]
            if np.isnan(np.sum(dummy)):
                lost_weight_inside += distro_w[i]

            else:
                signal += dummy
    print('We lost:', 100 * lost_weight_outside/np.sum(distro['weight']),
          'percent due to the limit of the smap')
    print('We lost:', 100 * lost_weight_inside/np.sum(distro['weight']),
          'due to fcol too low to fit resolutiosn')
    signal = np.reshape(signal, g_grid.shape)
    output = {
        'gyroradius': g_array,
        'pitch': p_array,
        'dgyr': g_array[1] - g_array[0],
        'dp': p_array[1] - p_array[0],
        'signal': signal,
        'lost_weight_inside': lost_weight_inside,
        'lost_weight_outside': lost_weight_outside
    }

    return output


def plot_synthetic_signal(r, p, signal, cmap=None, ax=None, fig=None,
                          ax_params={}, profiles=True, ax_profiles=None,
                          ax_params_profiles={}):
    """
    Plot the synthetic signal

    Jose Rueda: jrrueda@us.es

    :param  r: array with gyroradius values
    :param  p: array with pitch values
    :param  signal: matrix with the signal [np, nr]
    :param  cmap: color map to use, if none: ssplt.Gamma_II()
    :param  ax: axes where to plot, if none, a new figure will be created
    :param  ax_params: only used if the axis was created here, parameters for
    the axis_beauty function
    :param  profiles: flag to also plot the profiles
    :param  ax_profiles: axis where to draw the profiles, should be a list with
    the 2 axes
    :param  ax_params_profiles: only used if the axis was created here,
    parameters for the axis_beauty function, note, gyr and pitch labels are
    hardcored, sorry

    :return ax: axes where the figure was drawn
    """
    # Initialise the axis options:
    ax_options = {
        'xlabel': 'Pitch',
        'ylabel': 'Gyroradius [cm]',
    }
    ax_options.update(ax_params)

    ax_options_profiles = {
        'ylabel': 'Signal [a.u.]',
    }
    ax_options_profiles.update(ax_params_profiles)
    # Prepare the color map:
    if cmap is None:
        cmap = ssplt.Gamma_II()

    # Open the axis:
    if ax is None:
        fig, ax = plt.subplots()
        created = True
    else:
        created = False
    # plot the contour
    a1 = ax.contourf(p, r, signal.T, cmap=cmap)
    if created:
        ax = ssplt.axis_beauty(ax, ax_options)
        fig.colorbar(a1, ax=ax, label='Counts')
    fig.show()
    # Plot the profiles
    if ax_profiles is None:
        fig2, ax_profiles = plt.subplots(1, 2)
        created2 = True
    else:
        created2 = False
    ax_profiles[0].plot(p, np.sum(signal, axis=1))
    ax_profiles[1].plot(r, np.sum(signal, axis=0))
    if created2:
        ax_options_profiles['xlabel'] = 'Pitch [$\\degree$]'
        ax_profiles[0] = ssplt.axis_beauty(ax_profiles[0], ax_options_profiles)
        ax_options_profiles['xlabel'] = 'Gyroradius [cm]'
        ax_profiles[1] = ssplt.axis_beauty(ax_profiles[1], ax_options_profiles)
    fig2.show()
    return ax, ax_profiles


def synthetic_signal(pinhole_distribution: dict, efficiency, optics_parameters,
                     smap, scintillator, camera_parameters,
                     exp_time: float,
                     scint_synthetic_signal_params: dict = {},
                     spoints: str = None, px_shift: int = 0, py_shift: int = 0,
                     diag_parameters: dict = {},
                     noise_params: dict = {},
                     distortion_params: dict = {},
                     plot=True):
    """
    Calculate the camera synthetic signal of FILD detectors

    Jose Rueda: jrrueda@us.es

    Workflow:
        1- Calculate synthetic signal at the scintillator: photons/dgyr/dpitch
        2- relate gyroradius and pitch positions with pixels at the camera
        3- Map the scintillator distribution to the camera chip
        4- apply factors to consider transmission and finite focusing
    px_shift:-position of the 0,0
    Added in version 0.4.13
    """
    # --- Check inputs and initialise the settings
    # check/load strike map
    if isinstance(smap, str):  # if it is string, load the file.
        # if spoints is None:
        #     raise Exception('StrikePoints file needed!!')
        print('Reading strike map: ', smap)
        smap = ssmapping.StrikeMap(file=smap)
        smap.load_strike_points(spoints)
    if smap._resolutions is None:
        smap.calculate_phase_space_resolution(diag_params=diag_parameters)
    # check/load the scintillator
    if isinstance(scintillator, str):  # if it is string, load the file.
        print('Reading scintillator: ', scintillator)
        scintillator = ssmapping.Scintillator(scintillator)
    # check that the optics parameters included all necesary elements
    if 'beta' not in optics_parameters:
        print('Optic magnification not set, calculating proxy')
        xsize = camera_parameters['px_x_size'] * camera_parameters['nx']
        ysize = camera_parameters['px_y_size'] * camera_parameters['ny']
        chip_min_length = np.minimum(xsize, ysize)

        xscint_size = scintillator.coord_real['x1'].max() \
            - scintillator.coord_real['x1'].min()
        yscint_size = scintillator.coord_real['x1'].max() \
            - scintillator.coord_real['x2'].min()
        scintillator_max_length = np.maximum(xscint_size, yscint_size)

        beta = chip_min_length / scintillator_max_length
        print('Optics magnification, beta: ', beta)
    # Camera parameters:
    if isinstance(camera_parameters, str):
        camera_parameters = ssio.read_camera_properties(camera_parameters)
    # Grid for the synthetic signal at the scintillator
    scint_signal_options = {
        'rmin': 1.5,
        'rmax': 10.0,
        'dr': 0.1,
        'pmin': 20.0,
        'pmax': 80.0,
        'dp': 1.0,
    }
    scint_signal_options.update(scint_synthetic_signal_params)
    # Noise options:
    noise_options = {
        'dark_readout': {
            'apply': True
        },
        'camera_neutrons': {
            'percent': 0.001,
            'vmin': 0.7,
            'vmax': 0.9,
            'camera_bits': camera_parameters['range']
        },
        'broken': {
            'percent': 0.001
        },
        'photon': {
            'multiplier': 1.0
        },
        'ions': {
        },
        'betas': {
        },
        'neutrons': {
        },
        'gamma': {
        },
        'camera_gamma': {
        }
    }
    for i in noise_options.keys():
        if i in noise_params:
            noise_options[i].update(noise_params[i])
    # Distortion options
    distortion_options = {
        'model': 'WandImage',
        'parameters': {
            'method': 'barrel',
            'arguments': (0.2, 0.1, 0.1, 0.6)
        },
    }
    distortion_options.update(distortion_params)
    # Camera range:
    max_count = 2 ** camera_parameters['range'] - 1
    # --- Calculate the synthetic signal at the scintillator
    print(scint_signal_options)
    scint_signal = synthetic_signal_remap(pinhole_distribution, smap,
                                          efficiency=efficiency,
                                          **scint_signal_options)
    # --- Aling the center of the scintillator with the camera chip
    # center of the chip
    px_center = int(camera_parameters['ny'] / 2)
    py_center = int(camera_parameters['nx'] / 2)
    # center the scintillator at the coordinate origin
    y_scint_center = 0.5 * (scintillator._coord_real['x2'].max()
                            + scintillator._coord_real['x2'].min())
    x_scint_center = 0.5 * (scintillator._coord_real['x1'].max()
                            + scintillator._coord_real['x1'].min())
    scintillator._coord_real['x2'] -= y_scint_center
    scintillator._coord_real['x1']-= x_scint_center
    # shift the strike map by the same quantity:
    smap._data['x2'].data -= y_scint_center
    smap._data['x1'].data -= x_scint_center
    # center of the scintillator in pixel space
    px_0 = px_center + px_shift
    py_0 = py_center + py_shift
    # Scale to relate scintillator to camera
    xscale = optics_parameters['beta'] / camera_parameters['px_x_size']
    print('Xscale: ', xscale)
    yscale = optics_parameters['beta'] / camera_parameters['px_y_size']
    # calculate the pixel position of the scintillator vertices
    transformation_params = ssmapping.CalParams()
    transformation_params.xscale = xscale
    transformation_params.yscale = yscale
    transformation_params.xshift = px_0
    transformation_params.yshift = py_0
    scintillator.calculate_pixel_coordinates(transformation_params)
    # Align the strike map:
    smap.calculate_pixel_coordinates(transformation_params)
    smap.interp_grid((camera_parameters['nx'], camera_parameters['ny']),
                     MC_number=0)

    # --- Map scintillator and grid to frame
    n_gyr = scint_signal['gyroradius'].size
    n_pitch = scint_signal['pitch'].size
    synthetic_frame = np.zeros(smap._grid_interp['gyroradius'].shape)
    for ir in range(n_gyr):
        # Gyroradius limits to integrate
        gmin = scint_signal['gyroradius'][ir] - scint_signal['dgyr'] / 2.
        gmax = scint_signal['gyroradius'][ir] + scint_signal['dgyr'] / 2.
        for ip in range(n_pitch):
            # Pitch limits to integrates
            pmin = scint_signal['pitch'][ip] - scint_signal['dp'] / 2.
            pmax = scint_signal['pitch'][ip] + scint_signal['dp'] / 2.
            # Look for the pixels which cover this region:
            flags = (smap._grid_interp['gyroradius'] >= gmin) * (smap._grid_interp['gyroradius'] < gmax) \
                * (smap._grid_interp['pitch'] >= pmin) * (smap._grid_interp['pitch'] < pmax)
            flags = flags.astype(bool)
            # If there are some pixels, just divide the value weight among them
            n = np.sum(flags)
            if n > 0:
                synthetic_frame[flags] = scint_signal['signal'][ip, ir] / n \
                    * scint_signal['dgyr'] * scint_signal['dp']
    # --- Now apply all factors
    # Divide by 4\pi, ie, assume isotropic emission of the scintillator
    synthetic_frame *= 1 / 4 / np.pi
    print('Assuming isotropic emission of the scintillator light')
    # Consider the solid angle covered by the optics and the transmission of
    # the beam line through the lenses and mirrors:
    synthetic_frame *= optics_parameters['T'] * optics_parameters['Omega']
    # Photon to electrons in the camera sensor (QE)
    synthetic_frame *= camera_parameters['qe']
    # Electrons to counts in the camera sensor,
    synthetic_frame /= camera_parameters['ad_gain']
    # Consider the exposure time
    synthetic_frame *= exp_time
    # Pass to integer, as we are dealing with counts
    synthetic_frame = synthetic_frame.astype(int)
    original_frame = synthetic_frame.copy()
    # --- Add noise
    noise = {
        'dark_readout': None,
        'camera_neutrons': None,
        'broken': None,
        'photon': None,
        'ions': None,
        'betas': None,
        'neutrons': None,
        'gamma': None,
        'camera_gamma': None,
        'total': np.zeros(synthetic_frame.shape, int),
    }
    # broken pixels or fibers
    dummy = ssnoise.broken_pixels(synthetic_frame, **noise_options['broken'])
    noise['broken'] = dummy - synthetic_frame
    synthetic_frame = dummy
    # read out + dark noise:
    flag = ('dark_noise' in camera_parameters) and \
        ('readout_noise') in camera_parameters
    if flag:
        print('Including dark and reading noise')
        noise['dark_readout'] = \
            ssnoise.dark_plus_readout(synthetic_frame,
                                      camera_parameters['dark_noise'],
                                      camera_parameters['readout_noise'])
    else:
        noise['dark_readout'] = np.zeros(synthetic_frame.shape)
    noise['total'] += noise['dark_readout']
    # photon noise:
    print('Including photon noise')
    noise['photon'] = \
        ssnoise.photon(synthetic_frame, **noise_options['photon'])
    noise['total'] += noise['photon']
    # neutron noise:
    if plot:
        # just set the maximum of the scale for the future plot before the
        # neutrons saturate everything
        vmax = 0.95 * synthetic_frame.max()
    dummy = \
        ssnoise.camera_neutrons(synthetic_frame,
                                **noise_options['camera_neutrons'])

    noise['camera_neutrons'] = dummy - synthetic_frame
    synthetic_frame = dummy.copy()
    # Add the noise
    synthetic_frame += noise['total']
    # add the broken part to the noise frame, notice that we add it now because
    # this part of the noise was applied at the begining to the frame, as we
    # need to apply it before the photon noise is calculated (to do not
    # include the noise with \sqrt(N) to a pixel which do not have incident
    # photons due to a broken optical fiber, for example)
    noise['total'] += noise['broken']
    # idem with neutrons
    noise['total'] += noise['camera_neutrons']
    # --- Remove 'saturated' pixels:
    flags = synthetic_frame > max_count
    synthetic_frame[flags] = max_count
    synthetic_frame = synthetic_frame.astype(np.uint)
    flags = synthetic_frame < 0
    synthetic_frame[flags] = 0
    # --- Apply distortion
    distorted_frame = ssoptics.distort_image(synthetic_frame,
                                             distortion_options)
    output = {
        'noise': noise,
        'camera_frame': distorted_frame,
        'original_camera_frame_no_dist': synthetic_frame,
        'original_camera_frame': original_frame,
        'remap_frame': scint_signal
    }
    if plot:
        fig, ax = plt.subplots()
        img = ax.imshow(distorted_frame, cmap=ssplt.Gamma_II(), origin='lower',
                  vmin=0, vmax=255)
        smap.plot_pix(ax, labels=False)
        scintillator.plot_pix(ax)
        plt.colorbar(img)
        fig.show()
    return output


# -----------------------------------------------------------------------------
# --- Weight function
# -----------------------------------------------------------------------------
def build_weight_matrix(smap, rscint, pscint, rpin, ppin,
                        efficiency=None, spoints=None, diag_params: dict = {},
                        B=1.8, A=2.0, Z=1, only_gyroradius=False):
    """
    Build FILD weight function

    Jose Rueda Rueda: jrrueda@us.es

    Introduced in version 0.4.2

    :param  smap: Strike map object or path pointing to the strike map file


    :param  rscint: Gyroradius array in the scintillator grid
    :param  pscint: Pitch array in the scintillator grid
    :param  rpin: Gyroradius array in the pinhole grid
    :param  ppin: Pitch array in the pinhole grid
    :param  efficiency: ScintillatorEfficiency() object. If None, efficiency
    will not be included
    :param  spoints: path pointing to the strike point file. Not needed if smap
    is a strike map object with the resolutions already calculated
    :param  diag_params: Parametes for the resolution calculation, useless if
    the input strike map has the resolutions already calcualted See
    StrikeMap.calculate_resolutions() for the whole list of options
    :param  B: Magnetic field, used to translate between radius and energy, for
    the efficiency evaluation
    :param  A: Mass in amu, used to translate between radius and energy, for the
    efficiency evaluation
    :param  Z: charge in elecrton charges, used to translate between radius and
    energy, for the efficiency evaluation
    :param  only_gyroradius: flag to decide if the output will be the matrix
    just relating giroradius in the pinhole and the scintillator, ie, pitch
    integrated
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
        print('Considering scintillator efficiency in W')
    else:
        eff = np.ones(rpin.size)
    # Build the weight matrix. We will use brute force, I am sure that there is
    # a tensor product implemented in python which does the job in a more
    # efficient way
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

            col_factor = smap.interpolators['collimator_factor'](
                rpin[kk], ppin[ll]) / 100.0
            if col_factor > 0.0:
                # Calculate the contribution:
                dummy = col_factor * g_func(Rscint.flatten(), **g_parameters) \
                    * pitch_func(Pscint.flatten(), **p_parameters)\
                    * eff[kk]
                res_matrix[:, :, kk, ll] = np.reshape(dummy, Rscint.shape).T
            else:
                res_matrix[:, :, kk, ll] = 0.0
    res_matrix[np.isnan(res_matrix)] = 0.0

    # --- Collapse Weight function
    if not only_gyroradius:
        W2D = np.zeros((nr_scint * np_scint, nr_pin * np_pin))
        ## todo make this with an elegant numpy reshape, not manually
        print('Reshaping W... ')
        for irs in tqdm(range(nr_scint)):
            for ips in range(np_scint):
                for irp in range(nr_pin):
                    for ipp in range(np_pin):
                        W2D[irs * np_scint + ips, irp * np_pin + ipp] =\
                            res_matrix[irs, ips, irp, ipp]
    else:  # The required W2D is directly the integral of res_matrix
        W2D = np.sum(np.sum(res_matrix, axis=3), axis=1)
    return res_matrix, W2D


def plot_W(W4D, pr, pp, sr, sp, pp0: float = None, pr0: float = None,
           sp0: float = None, sr0: float = None,
           cmap=None, nlev=20):
    """
    Plot the weight function

    Jose Rueda Rueda: jrrueda@us.es

    :param  W4D: 4-D weight function
    :param  pr: array of gyroradius at the pinhole used to calculate W
    :param  pp: array of pitches at the pinhole used to calculate W
    :param  sr: array of gyroradius at the scintillator used to calculate W
    :param  sp: array of pitches at the scintillator used to calculate W
    :param  pp0: precise radius wanted at the pinhole to plot the scintillator W
    :param  pr0: precise pitch wanted at the pinhole to plot the scintillator W
    :param  sp0: precise radius wanted at the scintillator to plot the pinhole W
    :param  sr0: precise pitch wanted at the scintillator to plot the pinhole W

    Note, the pair (pr0, pp0) or (sr0, sp0) should be given, in this basic
    function they can't be mixed
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
        up = pp[ip]   # Used value of pitch
        ur = pr[ir]   # Used value of radius
        W = W4D[:, :, ir, ip]
        fig, ax = plt.subplots()
        a = ax.contourf(sp, sr, W, nlev, cmap=ccmap)
        plt.colorbar(a, ax=ax)
        ax.set_title('Pinhole point: (' + str(ur) + ', ' + str(up) + ')')
        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(sr, np.sum(W, axis=1))
        ax2[0].set_xlabel('$r_l$ [cm]')
        ax2[1].plot(sp, np.sum(W, axis=0))
        ax2[1].set_xlabel('Pitch')

    # Same but with the scintillator point
    if (sp0 is not None) and (sr0 is not None):
        ip = np.argmin(abs(sp - sp0))
        ir = np.argmin(abs(sr - sr0))
        up = sp[ip]   # Used value of pitch
        ur = sr[ir]   # Used value of radius
        W = W4D[ir, ip, :, :]
        fig, ax = plt.subplots()
        a = ax.contourf(pp, pr, W, nlev, cmap=ccmap)
        plt.colorbar(a, ax=ax)
        ax.set_title('Scint point: (' + str(ur) + ', ' + str(up) + ')')
        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].plot(pr, np.sum(W, axis=1))
        ax2[0].set_xlabel('$r_l$ [cm]')
        ax2[1].plot(pp, np.sum(W, axis=0))
        ax2[1].set_xlabel('Pitch')
