"""Remapping of FILD frames."""
import time
import os
import f90nml
import numpy as np
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
import Lib.SimulationCodes.SINPA as ssSINPA
import Lib.LibUtilities as ssextra
from Lib.LibMachine import machine
from Lib.LibMap.StrikeMap import StrikeMap
import Lib.LibMap.Common as common
import Lib.LibPaths as p
import Lib.LibIO as ssio
import Lib.errors as errors

from tqdm import tqdm   # For waitbars
paths = p.Path(machine)
del p


def remapAllLoadedFrames(video,
                         rmin: float = 1., rmax: float = 10.5, dr: float = 0.1,
                         pmin: float = 15., pmax: float = 90., dp: float = 1.0,
                         rprofmin: float = 1.5, rprofmax: float = 9.0,
                         pprofmin: float = 20.0, pprofmax: float = 90.0,
                         code_options: dict = {},
                         method: int = 1,
                         verbose: bool = False,
                         mask=None,
                         decimals: int = 1,
                         smap_folder: str = None,
                         map=None,
                         remap_method: str = 'centers',
                         MC_number: int = 100,
                         allIn: bool = False,
                         use_average: bool = False):
    """
    Remap all loaded frames from a FILD video.

    Jose Rueda Rueda: jrrueda@us.es

    @param    video: Video object (see LibVideoFiles)
    @param    rmin: minimum gyroradius to consider [cm]
    @param    rmax: maximum gyroradius to consider [cm]
    @param    dr: bin width in the radial direction
    @param    pmin: Minimum pitch to consider [º]
    @param    pmax: Maximum pitch to consider [º]
    @param    dp: bin width in pitch [º]
    @param    rprofmin: minimum gyrodarius to calculate pitch profiles [cm]
    @param    rprofmax: maximum gyroradius to calculate pitch profiles [cm]
    @param    pprofmin: minimum pitch for gyroradius profiles [º]
    @param    pprofmax: maximum pitch for gyroradius profiles [º]
    @param    code_options: namelist dictionary with the FILDSIM/SINPA options
              just in case we need to run the code. See FILDSIM/SINPA library
              and their gitlabs for the necesary options. It is recomended to
              leave this like {}, as the code will load the options used to
              generate the library, so the new calculated strike maps will be
              consistent with the old ones
    @param    verbose: Print the elapsed time
    @param    method: method to interpolate the strike maps, default 1: linear
    @param    decimals: Number of decimals to look for the strike map
    @param    smap_folder: Folder where to look for strike maps, if none
              the code will look in ...Suite/Data/RemapStrikeMaps/FILD/<geomID>
              with the <geomID> stored in the video object
    @param    map: Strike map to be used, if none, we will look in the folder
              for the right strike map
    @param    mask: binary mask defining the region of the scintillator we want
              to map. If it is a string pointing to a file, the mask saved in
              that file will be loaded
    @param    remap_method: 'MC' or 'centers', the method to be used for the
              remapping of the frames (MC recomended for tomography, but it
              needs 3 minutes per new strike map...)
    @param    number of MC markers for the MC remap
    @param    allIn: boolean flag to disconect the interaction with the user,
              where looking for the strike map in the database, we will take
              the closer one available in time, without expecting an answer for
              the user. This option was implemented to remap large number of
              shots 'automatically' without interaction from the user needed.
              Option not used if you give an input strike map
    @param    use_average: if true, use the averaged frames instead of the
              raw ones

    @return   output: dictionary containing all the outputs:
        -# 'frames': remaped_frames [xaxis(pitch), yaxis(r), taxis]
        -# 'xaxis': pitch,
        -# 'yaxis': gyr,
        -# 'xlabel': 'Pitch', label to plot
        -# 'ylabel': '$r_l$', label to plot
        -# 'xunits': '{}^o', units of the pitch
        -# 'yunits': 'cm', units of the gyroradius
        -# 'sprofx': signal integrated in gyroradius vs time
        -# 'sprofy': signal_integrated in pitch vs time
        -# 'sprofxlabel': label for sprofx
        -# 'sprofylabel': label for sprofy
        -# 'phi': phi, calculated phi angle, FILDSIM [deg]
        -# 'theta': theta, calculated theta angle FILDSIM [deg]
        -# 'theta_used': theta_used for the remap [deg]
        -# 'phi_used': phi_used for the remap [deg]
        -# 'tframes': time of the frames
        -# 'existing_smaps': array indicating which smaps where found in the
        database and which don't
    @return   opt: dictionary containing all the input parameters
    """
    # --- INPUTS CHECK AND PREPARATION
    # -- Check inputs strike map
    print('.-. . -- .- .--. .--. .. -. --.')
    if map is None:
        got_smap = False
    else:
        got_smap = True
        smap = map
        print('A StrikeMap was given, we will remap all frames with it')

    if smap_folder is None:
        smap_folder = os.path.join(paths.ScintSuite, 'Data', 'RemapStrikeMaps',
                                   'FILD', video.FILDgeometry)
    # -- Check which code generated the library
    if not got_smap:
        namelistFile = os.path.join(smap_folder, 'parameters.cfg')
        nml = f90nml.read(namelistFile)
        if 'n_pitch' in nml['config']:
            FILDSIM = True
        else:
            FILDSIM = False
    # -- Check the mask
    if type(mask) is str:
        # the user gave us a saved mask, not the matrix, so load the matrix:
        file = ssio.check_open_file(mask)
        [mask] = ssio.read_variable_ncdf(file, ['mask'], human=True)
        # tranform to bool
        mask = mask.astype(bool)
    # -- Check the tipe of remap
    if remap_method == 'centers':
        MC_number = 0  # to turn off the transformation matrix calculation
    # -- Prepare the frames
    if not use_average:
        data = video.exp_dat
    else:
        data = video.avg_dat
    # -- Get frame shape:
    nframes = data['frames'].shape[2]
    frame_shape = data['frames'].shape[0:2]
    # -- Get the time (to measure elapsed time)
    tic = time.time()
    # -- Prepare the grid
    ngyr = int((rmax-rmin)/dr) + 1
    npit = int((pmax-pmin)/dp) + 1
    p_edges = pmin - dp/2 + np.arange(npit+1) * dp
    g_edges = rmin - dr/2 + np.arange(ngyr+1) * dr
    gyr = 0.5 * (g_edges[0:-1] + g_edges[1:])
    pitch = 0.5 * (p_edges[0:-1] + p_edges[1:])

    # --- STRIKE MAP SEARCH
    exist = np.zeros(nframes, bool)
    name = ' '      # To save the name of the strike map
    name_old = ' '  # To avoid loading twice in a row the same map
    if not got_smap:
        # -- Collect the angles
        phi = video.Bangles['phi']
        theta = video.Bangles['theta']
        # Check that the angles were calculated for the frames (it can happen
        # that the user recalculate the angles after averaging, so they are
        # evaluated in the original time base)
        if (phi.size != nframes) or (theta.size != nframes):
            print('Number of frames: ', nframes)
            print('Size of phi: ', phi.size)
            print('Size of theta: ', theta.size)
            raise errors.NotValidInput('Wrong length of phi and theta')
        # -- See if the strike map exist in the folder
        print('Looking for strikemaps in: ', smap_folder)
        for iframe in tqdm(range(nframes)):
            if FILDSIM:
                name = ssFILDSIM.guess_strike_map_name_FILD(
                    phi[iframe], theta[iframe], geomID=video.FILDgeometry,
                    decimals=decimals
                    )
            else:
                name = ssSINPA.execution.guess_strike_map_name_FILD(
                    phi[iframe], theta[iframe], geomID=video.FILDgeometry,
                    decimals=decimals
                    )
            # See if the strike map exist
            if os.path.isfile(os.path.join(smap_folder, name)):
                exist[iframe] = True
        # -- See how many we need to calculate
        nnSmap = np.sum(~exist)  # Number of Smaps missing
        dummy = np.arange(nframes)     #
        existing_index = dummy[exist]  # Index of the maps we have
        non_existing_index = dummy[~exist]
        theta_used = np.round(theta, decimals=decimals)
        phi_used = np.round(phi, decimals=decimals)
        # The variable x will be the flag to calculate or not more strike maps
        if nnSmap == 0:
            print('--. .-. . .- -')
            print('Ideal situation, not a single map needs to be calculated')
        elif nnSmap == nframes:
            print('Non a single strike map, full calculation needed')
        elif nnSmap != 0:
            if not allIn:
                print('We need to calculate, at most:', nnSmap, 'StrikeMaps')
                print('Write 1 to proceed, 0 to take the closer'
                      + '(in time) existing strikemap')
                x = int(input('Enter answer:'))
            else:
                x = 0
            if x == 0:
                print('We will not calculate new strike maps')
                print('Looking for the closer ones')
                for i in tqdm(range(len(non_existing_index))):
                    ii = non_existing_index[i]
                    icloser = ssextra.find_nearest_sorted(existing_index, ii)
                    theta_used[ii] = theta[icloser]
                    phi_used[ii] = phi[icloser]
    else:  # the case the smap was passed as input
        theta = np.zeros(nframes)
        phi = np.zeros(nframes)
        exist = np.ones(nframes, bool)
        theta_used = np.zeros(nframes, bool)
        phi_used = np.zeros(nframes, bool)
    # --- REMAP THE FRAMES
    # -- Initialise the variables:
    remaped_frames = np.zeros((npit, ngyr, nframes))
    signal_in_gyr = np.zeros((ngyr, nframes))
    signal_in_pit = np.zeros((npit, nframes))
    print('Remapping frames ...')
    for iframe in tqdm(range(nframes)):
        if not got_smap:
            if FILDSIM:
                name = ssFILDSIM.find_strike_map(
                    phi_used[iframe], theta_used[iframe], smap_folder,
                    geomID=video.FILDgeometry, FILDSIM_options=code_options,
                    decimals=decimals, clean=True)
            else:  # SINPA CODE
                name = ssSINPA.execution.find_strike_map_FILD(
                    phi_used[iframe], theta_used[iframe], smap_folder,
                    geomID=video.FILDgeometry, SINPA_options=code_options,
                    decimals=decimals, clean=True)
        # Only reload the strike map if it is needed
        if name != name_old:
            smap = StrikeMap(0, os.path.join(smap_folder, name))
            smap.calculate_pixel_coordinates(video.CameraCalibration)
            # print('Interpolating grid')
            smap.interp_grid(frame_shape, plot=False, method=method,
                             MC_number=MC_number,
                             grid_params={'ymin': rmin, 'ymax': rmax,
                                          'dy': dr,
                                          'xmin': pmin, 'xmax': pmax,
                                          'dx': dp})
        name_old = name
        # remap the frames
        remaped_frames[:, :, iframe] = \
            common.remap(smap, data['frames'][:, :, iframe],
                         x_edges=p_edges, y_edges=g_edges, mask=mask,
                         method=remap_method)
        # Calculate the gyroradius and pitch profiles
        dummy = remaped_frames[:, :, iframe].squeeze()
        signal_in_gyr[:, iframe] = common.gyr_profile(dummy, pitch, pprofmin,
                                                      pprofmax)
        signal_in_pit[:, iframe] = common.pitch_profile(dummy, gyr, rprofmin,
                                                        rprofmax)
    if verbose:
        toc = time.time()
        print('Whole time interval remapped in: ', toc-tic, ' s')
        print('Average time per frame: ', (toc-tic) / nframes, ' s')

    output = {
        'frames': remaped_frames,
        'xaxis': pitch, 'yaxis': gyr,
        'xlabel': 'Pitch', 'ylabel': '$r_l$',
        'xunits': '$\\degree$', 'yunits': 'cm',
        'sprofx': signal_in_pit, 'sprofy': signal_in_gyr,
        'sprofxlabel': 'Signal integrated in $r_l$',
        'sprofylabel': 'Signal integrated in pitch',
        'phi': phi, 'theta': theta,
        'theta_used': theta_used, 'phi_used': phi_used,
        'nframes': video.exp_dat['nframes'],
        'tframes': video.exp_dat['tframes'],
        'existing_smaps': exist
    }
    opt = {
        'rmin': rmin, 'rmax': rmax, 'dr': dr, 'pmin': pmin, 'pmax': pmax,
        'dp': dp, 'rprofmin': rprofmin, 'rprofmax': rprofmax,
        'pprofmin': pprofmin, 'pprofmax': pprofmax,
        'decimals': decimals,
        'smap_folder': smap_folder,
        'use_average': use_average,
        'CodeUsed': smap.code,
    }
    return output, opt
