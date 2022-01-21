"""Remapping of FILD frames."""
import time
import os
import numpy as np
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
import Lib.LibUtilities as ssextra
from Lib.LibMachine import machine
from Lib.LibMap.StrikeMap import StrikeMap
import Lib.LibMap.Common as common
import Lib.LibPaths as p
import Lib.LibIO as ssio
import Lib.LibData as ssdat
from tqdm import tqdm   # For waitbars
paths = p.Path(machine)
del p


def remap_all_loaded_frames(video, calibration, shot, rmin: float = 1.0,
                            rmax: float = 10.5, dr: float = 0.1,
                            pmin: float = 15.0, pmax: float = 90.0,
                            dp: float = 1.0, rprofmin: float = 1.5,
                            rprofmax: float = 9.0, pprofmin: float = 20.0,
                            pprofmax: float = 90.0, rfild: float = 2.186,
                            zfild: float = 0.32, alpha: float = 0.0,
                            beta: float = -12.0,
                            fildsim_options: dict = {},
                            method: int = 1,
                            verbose: bool = False, mask=None,
                            machine: str = 'AUG',
                            decimals: int = 1,
                            smap_folder: str = None,
                            map=None,
                            remap_method: str = 'centers',
                            MC_number: int = 100,
                            allIn: bool = False):
    """
    Remap all loaded frames from a FILD video.

    Jose Rueda Rueda: jrrueda@us.es

    @param    video: Video object (see LibVideoFiles)
    @param    calibration: Calibation object (see Calibration class)
    @param    shot: shot number
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
    @param    rfild: Radial position of FILD [m]
    @param    zfild: height avobe the mid plane of FILD head [m]
    @param    alpha: Alpha orientation of FILD head [º]
    @param    beta: beta orientation of FILD head [º]
    @param    fildsim_options: namelist dictionary with the fildsim options
              just in case we need to run FILDSIM. See FILDSIM library and
              FILDSIM gitlab for the necesary options. It is recomended to
              leave this like {}, as the code will load the options used to
              generate the library, so the new calculated strike maps will be
              consistent with the old ones
    @param    verbose: Print the elapsed time
    @param    machine: name of the machine, to guess the name of the strike map
              to be used. Notice that this option will be deprecated in a near
              future, in the new versions, each FILD will have its database
    @param    method: method to interpolate the strike maps, default 1: linear
    @param    decimals: Number of decimals to look for the strike map
    @param    smap_folder: folder where to look for strike maps, if none, the
              code will use the indicated by LibPaths. This will change in
              future versions. As the folder will be indicated in the FILD info
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
        -# 'bfield': Modulus of the field vs time [T]
        -# 'phi': phi, calculated phi angle, FILDSIM [deg]
        -# 'theta': theta, calculated theta angle FILDSIM [deg]
        -# 'theta_used': theta_used for the remap [deg]
        -# 'phi_used': phi_used for the remap [deg]
        -# 'tframes': time of the frames
        -# 'existing_smaps': array indicating which smaps where found in the
        database and which don't
    @return   opt: dictionary containing all the input parameters
    """
    # Check inputs strike map
    print('.-. . -- .- .--. .--. .. -. --.')
    if map is None:
        got_smap = False
    else:
        got_smap = True
        print('A StrikeMap was given, we will remap all frames with it')

    if smap_folder is None:
        smap_folder = paths.FILDStrikeMapsRemap

    if type(mask) is str:
        # the user gave us a saved mask, not the matrix, so load the matrix:
        file = ssio.check_open_file(mask)
        [mask] = ssio.read_variable_ncdf(file, ['mask'], human=True)
        # tranform to bool
        mask = mask.astype(np.bool)
    # Print  some info:
    if not got_smap:
        print('Looking for strikemaps in: ', smap_folder)
    # Check the tipe of remap
    if remap_method == 'centers':
        MC_number = 0  # to turn off the transformation matrix calculation
    # Get frame shape:
    nframes = len(video.exp_dat['nframes'])
    frame_shape = video.exp_dat['frames'].shape[0:2]

    # Get the time (to measure elapsed time)
    tic = time.time()

    # Get the magnetic field: In principle we should be able to do this in an
    # efficient way, but the AUG library to access magnetic field is kind of a
    # shit in python 3, so we need a work around
    if not got_smap:
        if machine == 'AUG':
            print('Opening shotfile from magnetic field')
            import map_equ as meq
            equ = meq.equ_map(shot, diag='EQH')
        br = np.zeros(nframes)
        bz = np.zeros(nframes)
        bt = np.zeros(nframes)
    b_field = np.zeros(nframes)
    # br, bz, bt, bp =\
    #     ssdat.get_mag_field(shot, rfild, zfild,
    #                         time=video.exp_dat['tframes'])
    # Get the modulus of the field
    # b_field = np.sqrt(br**2 + bz**2 + bt**2)
    # Initialise the variables:
    ngyr = int((rmax-rmin)/dr) + 1
    npit = int((pmax-pmin)/dp) + 1
    remaped_frames = np.zeros((npit, ngyr, nframes))
    signal_in_gyr = np.zeros((ngyr, nframes))
    signal_in_pit = np.zeros((npit, nframes))
    theta = np.zeros(nframes)
    phi = np.zeros(nframes)
    name_old = ' '
    name = ' '
    exist = np.zeros(nframes, np.bool)
    # --- Calculate the grid
    p_edges = pmin - dp/2 + np.arange(npit+1) * dp
    g_edges = rmin - dr/2 + np.arange(ngyr+1) * dr
    gyr = 0.5 * (g_edges[0:-1] + g_edges[1:])
    pitch = 0.5 * (p_edges[0:-1] + p_edges[1:])
    # --- Calculate the theta and phi angles
    if not got_smap:  # if no smap was given calculate the theta and phi
        print('Calculating theta and phi')
        for iframe in tqdm(range(nframes)):
            if machine == 'AUG':
                tframe = video.exp_dat['tframes'][iframe]
                br[iframe], bz[iframe], bt[iframe], bp =\
                    ssdat.get_mag_field(shot, rfild, zfild, time=tframe,
                                        equ=equ)
                b_field[iframe] = np.sqrt(br[iframe]**2 + bz[iframe]**2
                                          + bt[iframe]**2)
            phi[iframe], theta[iframe] = \
                ssFILDSIM.calculate_fild_orientation(br[iframe], bz[iframe],
                                                     bt[iframe], alpha, beta)
            name = ssFILDSIM.guess_strike_map_name_FILD(phi[iframe],
                                                        theta[iframe],
                                                        machine=machine,
                                                        decimals=decimals)
            # See if the strike map exist
            if os.path.isfile(os.path.join(smap_folder, name)):
                exist[iframe] = True
    else:
        exist = np.ones(nframes, np.bool)
        theta_used = np.zeros(nframes, np.bool)
        phi_used = np.zeros(nframes, np.bool)
    # See how many strike maps we need to calculate:
    if not got_smap:
        nnSmap = np.sum(~exist)  # Number of Smaps missing
        dummy = np.arange(nframes)     #
        existing_index = dummy[exist]  # Index of the maps we have
        non_existing_index = dummy[~exist]
        theta_used = np.round(theta, decimals=decimals)
        phi_used = np.round(phi, decimals=decimals)

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
    print('Remapping frames ...')
    for iframe in tqdm(range(nframes)):
        if not got_smap:
            name = ssFILDSIM.find_strike_map(rfild, zfild, phi_used[iframe],
                                             theta_used[iframe], smap_folder,
                                             machine=machine,
                                             FILDSIM_options=fildsim_options,
                                             decimals=decimals)
        # Only reload the strike map if it is needed
        if name != name_old:
            map = StrikeMap(0, os.path.join(smap_folder, name))
            map.calculate_pixel_coordinates(calibration)
            # print('Interpolating grid')
            map.interp_grid(frame_shape, plot=False, method=method,
                            MC_number=MC_number,
                            grid_params={'ymin': rmin, 'ymax': rmax,
                                         'dy': dr,
                                         'xmin': pmin, 'xmax': pmax,
                                         'dx': dp})
        name_old = name
        # remap the frames
        remaped_frames[:, :, iframe] = \
            common.remap(map, video.exp_dat['frames'][:, :, iframe],
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
    output = {'frames': remaped_frames, 'xaxis': pitch, 'yaxis': gyr,
              'xlabel': 'Pitch', 'ylabel': '$r_l$',
              'xunits': '$\\degree$', 'yunits': 'cm',
              'sprofx': signal_in_pit, 'sprofy': signal_in_gyr,
              'sprofxlabel': 'Signal integrated in r_l',
              'sprofylabel': 'Signal integrated in pitch',
              'bfield': b_field, 'phi': phi, 'theta': theta,
              'theta_used': theta_used, 'phi_used': phi_used,
              'nframes': video.exp_dat['nframes'],
              'tframes': video.exp_dat['tframes'],
              'existing_smaps': exist}
    opt = {'rmin': rmin, 'rmax': rmax, 'dr': dr, 'pmin': pmin, 'pmax': pmax,
           'dp': dp, 'rprofmin': rprofmin, 'rprofmax': rprofmax,
           'pprofmin': pprofmin, 'pprofmax': pprofmax, 'rfild': rfild,
           'zfild': zfild, 'alpha': alpha, 'beta': beta,
           'calibration': calibration, 'decimals': decimals,
           'smap_folder': smap_folder}
    return output, opt
