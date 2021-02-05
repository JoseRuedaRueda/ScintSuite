"""
Contains the routines to perform the tomographic inversion

NOTICE: I consider the creatation of the transfer matrix as an issue of the
synthetic codes (INPASIM, FILDSIM, i/HIBPSIM) therefore the routines which
create these matrix are placed are their corresponding libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import LibFILDSIM as ssfildsim
import LibMap as ssmapping
from scipy import ndimage        # To denoise the frames
from tqdm import tqdm            # For waitbars
# -----------------------------------------------------------------------------
# SOLVERS AND REGRESSION ALGORITHMS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FILD TOMOGRAPHY
# -----------------------------------------------------------------------------
def prepare_X_y_FILD(frame, smap, s_opt: dict, p_opt: dict,
                     verbose: bool = True, plt_frame: bool = False,
                     LIMIT_REGION_FCOL: bool = True, denoise: bool = True,
                     efficiency: dict = {}):
    """
    Prepare the arrays to perform the tomographic inversion in FILD

    Jose Rueda: jrrueda@us.es


    @param    frame: camera frame (in photons /s) you can put directly the
    camera frame if you want a.u
    @param    smap: Scintillator map, with resolution calculated
    @param    s_opt: Parameters for the scintillator grid
    @param    p_opt: Parameters for the pinhole grid
    @param    verbose: Print some notes on the console
    @param    plt_frame: Plot the frame and noise suppresed frame (todo)
    @param    LIMIT_REGION_FCOL: Limit the pinhole grid to points with fcol>0
    @param    denoise: apply median filter to the frame
    @param    efficiency: efficiecy dictionary
    @return   signal1D:  Signal filtered and reduced in 1D array
    @return   W2D: Weight function compresed as 2D
    """
    print('.--. ... ..-. -')
    print('Preparing W and the measurement')
    # --- create the grids
    redges = np.arange(start=s_opt['rmin'], stop=s_opt['rmax'],
                       step=s_opt['dr'])
    pedges = np.arange(start=s_opt['pmin'], stop=s_opt['pmax'],
                       step=s_opt['dp'])
    scint_grid = {'nr': len(redges) - 1, 'np': len(pedges) - 1,
                  'r': 0.5 * (redges[0:-1] + redges[1:]),
                  'p': 0.5 * (pedges[0:-1] + pedges[1:])}

    redges = np.arange(start=p_opt['rmin'], stop=p_opt['rmax'],
                       step=p_opt['dr'])
    pedges = np.arange(start=p_opt['pmin'], stop=p_opt['pmax'],
                       step=p_opt['dp'])
    pin_grid = {'nr': len(redges) - 1, 'np': len(pedges) - 1,
                'r': 0.5 * (redges[0:-1] + redges[1:]),
                'p': 0.5 * (pedges[0:-1] + pedges[1:])}
    # --- Denoise the frame with the median filter
    if denoise:
        frame2 = ndimage.median_filter(frame, 3)
        if verbose:
            print('Median filter applied to the frame')
            print('Previous max: ', frame.max(), ' min: ', frame.min())
            print('New max: ', frame2.max(), ' min: ', frame2.min())
    # --- Remap the frame
    rep_frame, r, p = ssmapping.remap(smap, frame2, x_min=s_opt['pmin'],
                                      x_max=s_opt['pmax'],
                                      delta_x=s_opt['dp'],
                                      y_min=s_opt['rmin'],
                                      y_max=s_opt['rmax'],
                                      delta_y=s_opt['dr'])
    rep_frame = rep_frame.T

    # --- Limit the grid
    if LIMIT_REGION_FCOL:
        print('Grid Definition --> Limiting to regions where FCOL>0')
        # Find gyr and pitch with fcol>0
        flags = smap.collimator_factor > 0.
        dummy_r = smap.gyroradius[flags]
        dummy_p = smap.pitch[flags]
        unique_r = np.unique(dummy_r)
        unique_p = np.unique(dummy_p)
        minr = unique_r.min()
        maxr = unique_r.max()
        minp = unique_p.min()
        maxp = unique_p.max()
        # Select only those values on the scint grid
        flags_r = (pin_grid['r'] > minr) * (pin_grid['r'] < maxr)
        flags_p = (pin_grid['p'] > minp) * (pin_grid['p'] < maxp)

        pgrid = {'nr': np.sum(flags_r), 'np': np.sum(flags_p),
                 'r': pin_grid['r'][flags_r], 'p': pin_grid['p'][flags_p]}
    else:
        pgrid = pin_grid

    if verbose:
        print('---')
        print('PINHOLE GRID')
        print(pgrid['nr'], ' x ', pgrid['np'])
        print('Gyroradius (cm):', pgrid['r'].min(), pgrid['r'].max())
        print('Pitch Angle (degree)', pgrid['p'].min(), pgrid['p'].max())
        print('---')
        print('SCINTILLATOR GRID')
        print(scint_grid['nr'], ' x ', scint_grid['np'])
        print('Gyroradius (cm):', scint_grid['r'].min(), scint_grid['r'].max())
        print('Pitch Angle (degree)',
              scint_grid['p'].min(), scint_grid['p'].max())
        print('---')

    # --- Build transfer function
    W4D = ssfildsim.build_weight_matrix(smap, scint_grid['r'], scint_grid['p'],
                                        pgrid['r'], pgrid['p'], efficiency)

    # --- Collapse Weight function
    W2D = np.zeros((scint_grid['nr'] * scint_grid['np'],
                   pgrid['nr'] * pgrid['np']))
    ## todo make this with an elegant numpy reshape, not manually
    print('Reshaping W: ')
    for irs in tqdm(range(scint_grid['nr'])):
        for ips in range(scint_grid['np']):
            for irp in range(pgrid['nr']):
                for ipp in range(pgrid['np']):
                    W2D[irs * scint_grid['np'] + ips, irp * pgrid['np'] + ipp]\
                        = W4D[irs, ips, irp, ipp]
    # --- Collapse signal into 1D
    signal1D = np.zeros(scint_grid['nr'] * scint_grid['np'])
    print(signal1D.shape)
    print(rep_frame.T.shape)
    print(scint_grid['nr'])
    print(scint_grid['np'])
    for irs in range(scint_grid['nr']):
        for ips in range(scint_grid['np']):
            signal1D[irs * scint_grid['np'] + ips] = rep_frame[irs, ips]

    return signal1D, W2D, W4D, scint_grid, pgrid
