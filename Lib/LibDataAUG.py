"""Routines to interact with the AUG database"""

import dd                # Module to load shotfiles
import get_gc            # Module to load vessel components
import numpy as np
import map_equ as meq    # Module to map the equilibrium
import os
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import warnings
from tqdm import tqdm
from scipy.interpolate import interpn, interp1d, interp2d
from LibPaths import Path
from warnings import *
pa = Path()


# -----------------------------------------------------------------------------
# --- AUG parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001

## Field and current direction
## @todo> This is hardcored here, at the end there are only 2 weeks of reverse
# field experiments in  the whole year, but if necesary, we could include here
# some kind of method to check the sign calling the AUG database
Bt_sign = 1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = -1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from the paper:
# J. Ayllon-Guerola et al. 2019 JINST14 C10032
# betas are taken to be -12.0 for AUG
# @todo include jet FILD as 'fild6', Jt-60 as 'fild7' as MAST-U as 'fild8'?
fild1 = {'alpha': 0.0,   # Alpha angle [deg], see paper
         'beta': -12.0,  # beta angle [deg], see FILDSIM doc
         'sector': 8,    # The sector where FILD is located
         'r': 2.180,     # Radial position [m]
         'z': 0.3,       # Z position [m]
         'phi_tor': 169.75,  # Toroidal position, [deg]
         'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
         'camera': 'PHANTOM',  # Type of used camera
         'extension': '_v710.cin',  # Extension of the video file, none for png
         'label': 'FILD1',  # Label for the diagnostic, for FILD6 (rFILD)
         'diag': 'FHC',  # name of the diagnostic for the fast channel
         'channel': 'FILD3_',  # prefix of the name of each channel (shotfile)
         'nch': 20}  # Number of fast channels

fild2 = {'alpha': 0.0, 'beta': -12.0, 'sector': 3, 'r': 2.180,
         'z': 0.3, 'phi_tor': 57.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
         'extension': '', 'label': 'FILD2', 'diag': 'FHA', 'channel': 'FIPM_',
         'nch': 20, 'camera': 'CCD'}

fild3 = {'alpha': 72.0, 'beta': -12.0, 'sector': 13, 'r': 1.975,
         'z': 0.765, 'phi_tor': 282.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD3/',
         'extension': '', 'label': 'FILD3', 'diag': 'xxx', 'channel': 'xxxxx',
         'nch': 99, 'camera': 'CCD'}

fild4 = {'alpha': 0.0, 'beta': -12.0, 'sector': 8, 'r': 2.035,
         'z': -0.462, 'phi_tor': 169.75,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD4/',
         'extension': '', 'label': 'FILD4', 'diag': 'FHD', 'channel': 'Chan-',
         'nch': 32, 'camera': 'CCD'}

fild5 = {'alpha': -48.3, 'beta': -12.0, 'sector': 7, 'r': 1.772,
         'z': -0.798, 'phi_tor': 147.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD5/',
         'extension': '', 'label': 'FILD5', 'diag': 'FHE', 'channel': 'Chan-',
         'nch': 64, 'camera': 'CCD'}

fild6 = {'alpha': 0.0, 'beta': 171.3, 'sector': 8, 'r': 2.180,
         'z': 0.3, 'phi_tor': 169.75,
         'path': '/p/IPP/AUG/rawfiles/FIT/',
         'extension': '_v710.cin', 'label': 'RFILD',
         'diag': 'FHC', 'channel': 'FILD3_', 'nch': 20, 'camera': 'CCD'}

FILD = (fild1, fild2, fild3, fild4, fild5, fild6)
## FILD diag names:
# fast-channels:
fild_diag = ['FHC', 'FHA', 'XXX', 'FHD', 'FHE', 'FHC']
fild_signals = ['FILD3_', 'FIPM_', 'XXX', 'Chan-', 'Chan-', 'FILD3_']
fild_number_of_channels = [20, 20, 99, 32, 64, 20]

# -----------------------------------------------------------------------------
# --- Magnetics data.
# -----------------------------------------------------------------------------
mag_coils_grp2coilName = { 'C07': ['C07', np.arange(1, 32)],
                           'C09': ['C07', np.arange(1, 32)],
                           'B-31_5_11': ['B31', np.arange(5, 1)],
                           'B-31_32_27' :['C07', np.arange(32, 38)]
                            }

# -----------------------------------------------------------------------------
# --- Equilibrium and magnetic field
# -----------------------------------------------------------------------------
def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, equ=None):
    """
    Wrapp to get AUG magnetic field

    Jose Rueda: jrrueda@us.es

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @return br: Radial magnetic field (nt, nrz_in), [T]
    @return bz: z magnetic field (nt, nrz_in), [T]
    @return bt: toroidal magnetic field (nt, nrz_in), [T]
    @return bp: poloidal magnetic field (nt, nrz_in), [T]
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True
    # Now calculate the field
    br, bz, bt = equ.rz2brzt(Rin, zin, t_in=time)
    bp = np.hypot(br, bz)
    # If we opened the equilibrium object, let's close it
    if created:
        equ.Close()
    return br, bz, bt, bp


def get_rho(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
            ed: int = 0, time: float = None, equ=None,
            coord_out: str = 'rho_pol'):
    """
    Wrap to get AUG magnetic field

    Jose Rueda: jrrueda@us.es

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @param coord_out: the desired rho coordinate, default rho_pol
    @return rho: The desired rho coordinate evaluated at the points
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True
    # Now calculate the field
    rho = equ.rz2rho(Rin, zin, t_in=time, coord_out=coord_out,
                     extrapolate=True)
    # If we opened the equilibrium object, let's close it
    if created:
        equ.Close()
    return rho


def get_psipol(shot: int, Rin, zin, diag='EQH', exp: str = 'AUGD',
               ed: int = 0, time: float = None, equ=None):
    """
    Wrap to get AUG poloidal flux field

    Jose Rueda: jrrueda@us.es
    ft.
    Pablo Oyola: pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @return psipol: Poloidal flux evaluated in the input grid.
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True

    equ.read_pfm()
    i = np.argmin(np.abs(equ.t_eq - time))
    PFM = equ.pfm[:, :, i].squeeze()
    psipol = interpn((equ.Rmesh, equ.Zmesh), PFM, (Rin, zin))

    # If we opened the equilibrium object, let's close it
    if created:
        equ.Close()

    return psipol


def get_shot_basics(shotnumber: int = None, diag: str = 'EQH', 
                    exp: str = 'AUGD', edition: int = 0,
                    time: float = None):
    """
    Retrieves from the equilibrium reconstruction the basic data stored into 
    a dictionary. Technically, it reads the SSQ from the equilibrium 
    diagnostic.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param shot: Shot number.
    @param diag: Equilibrium diagnostic. By default EQH.
    @param exp: Experiment where the data is stored.
    @param edition: Edition of the shotfile.
    @param time: time interval to retrieve. If it is a single value, only the
    appropriate data point will be retrieved. If None, all the data points are
    obtained.
    """
    
    # Checking the inputs.
    new_equ_opened = False
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp, edition=edition)
        new_equ_opened = True
    except:
        raise Exception('EQU shotfile cannot be opened.')
        
        
    # Deactivate the nasty warnings for a while.
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    eqh_time = np.asarray(sf(name='time').data) # Time data.
    
    # Checking the time data.
    nt = len(eqh_time)
    if time is None:
        t0 = 0
        t1 = nt
    elif len(time) == 1:
        t0 = np.abs(eqh_time.flatten() - time).argmin()
        t1 = t0
    else:
        t0 = np.abs(eqh_time.flatten() - time[0]).argmin() - 1
        t1 = np.abs(eqh_time.flatten() - time[-1]).argmin() + 1
    
    # Getting the names and the SSQ data.
    eqh_ssqnames = sf.GetSignal(name='SSQnam')
    eqh_ssq = sf.GetSignal(name='SSQ')
    warnings.filterwarnings('default') 
    
    # Unpacking the data.
    ssq = dict()
    for jssq in range(eqh_ssq.shape[1]):
        tmp = b''.join(eqh_ssqnames[jssq,:]).strip()
        lbl = tmp.decode('utf8')
        if lbl.strip() != '':
            ssq[lbl] = eqh_ssq[t0:t1, jssq]
            
    if new_equ_opened:
        sf.close()
    
    # Adding the time.
    ssq['time'] = eqh_time[t0:t1]
    
    #--- Reading the plasma current.
    try:
        sf = dd.shotfile(pulseNumber=shotnumber, diagnostic='MAG',
                          experiment='AUGD', edition=0)
    except:
        raise Exception('Error loading the MAG shotfile')
    
    # Getting the raw data.
    ipa_raw = sf(name='Ipa', tBegin=ssq['time'][0], tEnd=ssq['time'][-1])
    ipa     = ipa_raw.data
    ipa_time= ipa_raw.time
    
    # Getting the calibration.
    multi = sf.getParameter('06ULID12', 'MULTIA00').data.astype(dtype=float)
    shift = sf.getParameter('06ULID12', 'SHIFTB00').data.astype(dtype=float)
    
    ssq['ip'] = ipa*multi + shift # This provides the current in A.
    ssq['ip'] *= 1.0e-6
    ssq['iptime'] = ipa_time
    
    # Close the shotfile.
    sf.close()
    
    #--- Getting the magnetic field at the axis.
    try:
        sf = dd.shotfile(pulseNumber=shotnumber, experiment='AUGD', 
                         diagnostic='MAI', edition=0)
    except:
        raise Exception('MAI shotfile could not be loaded!')
        
    # Getting toroidal field.
    btf_sf   = sf(name='BTF', tBegin=ssq['time'][0], tEnd=ssq['time'][-1])
    btf      = btf_sf.data
    btf_time = btf_sf.time
    
    # Getting the calibration.
    multi = sf.getParameter('14BTF', 'MULTIA00').data.astype(dtype=float)
    shift = sf.getParameter('14BTF', 'SHIFTB00').data.astype(dtype=float)
    
    ssq['bt0'] = multi*btf + shift
    ssq['bttime'] = btf_time
            
    return ssq

# -----------------------------------------------------------------------------
# --- Electron density and temperature profiles.
# -----------------------------------------------------------------------------
def get_ne(shotnumber: int, time: float, exp: str = 'AUGD', diag: str = 'IDA',
           edition: int = 0):
    """
    Wrap to get AUG electron density.

    Pablo Oyola: pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param time: Time point to read the profile.
    @param exp: Experiment name.
    @param diag: diagnostic from which 'ne' will extracted.
    @param edition: edition of the shotfile to be read.

    @return output: a dictionary containing the electron density evaluated
    in the input times and the corresponding rhopol base.
    """
    # --- Opening the shotfile.
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp, edition=edition)
    except:
        raise NameError('The shotnumber %d is not in the database'%shotnumber)

    # --- Reading from the database
    ne = sf(name='ne')

    # The area base is usually constant.
    rhop = ne.area.data[0, :]

    # Getting the time base since for the IDA shotfile, the whole data
    # is extracted at the time.
    timebase = sf(name='time')

    # Making the grid.
    TT, RR = np.meshgrid(time, rhop)

    # Interpolating in time to get the input times.
    ne_out = interpn((timebase, rhop), ne.data, (TT.flatten(), RR.flatten()))

    ne_out = ne_out.reshape(RR.shape)

    # Output dictionary:
    output = {'data': ne_out, 'rhop': rhop}

    # --- Closing the shotfile.
    sf.close()

    return output


def get_Te(shotnumber: int, time: float, exp: str = 'AUGD', diag: str = 'CEZ',
           edition: int = 0):
    """
    Wrap to get AUG ion temperature.

    Pablo Oyola: pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param time: Time point to read the profile.
    @param exp: Experiment name.
    @param diag: diagnostic from which 'Te' will extracted.
    @param edition: edition of the shotfile to be read.

    @return output: a dictionary containing the electron temp. evaluated
    in the input times and the corresponding rhopol base.
    """
    # --- Opening the shotfile.
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp, edition=edition)
    except:
        raise NameError('The shotnumber %d is not in the database'%shotnumber)

    # --- Reading from the database
    te = sf(name='Te')

    # The area base is usually constant.
    rhop = te.area.data[0, :]

    # Getting the time base since for the IDA shotfile, the whole data
    # is extracted at the time.
    timebase = sf(name='time')

    # Making the grid.
    TT, RR = np.meshgrid(time, rhop)

    # Interpolating in time to get the input times.
    te_out = interpn((timebase, rhop), te.data,
                     (TT.flatten(), RR.flatten()))

    te_out = te_out.reshape(RR.shape)
    # Output dictionary:
    output = {'data': te_out, 'rhop': rhop}

    sf.close()

    return output


# -----------------------------------------------------------------------------
# --- ECE data.
# -----------------------------------------------------------------------------
def get_ECE(shotnumber: int, timeWindow: float = None, fast: bool = False,
            rhopLimits: float = None):
    """
    Retrieves from the database the ECE data and the calibrations to obtain
    the electron temperature perturbations.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param shotnumber: shot Number to get the ECE data.
    @param timeWindow: time window to get the data.
    @param fast: fast reading implies that the position of each of the
    radiometers is averaged along the time window making it faster.
    @param rhopLimits: limits in rho poloidal to retrieve the ECE. If None,
    the limits are set to [-1.0, 1.0], where the sign is conventional to 
    HFS (-) and LFS (+)
    """
    
    #--- Opening the shotfiles.
    exp = 'AUGD' # Only the AUGD data retrieve is supported with the dd.
    dt = 2*5.0e-3  # To get a bit more on the edges to make a better interp.
    method = 'linear' # Interpolation method
    try:
        sf_cec = dd.shotfile(diagnostic='CEC', pulseNumber=shotnumber,
                         experiment=exp,  edition=0) # Slow channel - ECE data.
        sf_rmc = dd.shotfile(diagnostic='RMC', pulseNumber=shotnumber,
                         experiment=exp,  edition=0) # Fast channel - ECE data.
    except:
        raise Exception('Shotfile not existent for ECE/RMD/RMC')
        
    try:
        eqh = dd.shotfile(diagnostic='EQH', pulseNumber=shotnumber,
                          experiment=exp, edition=0)
        equ = meq.equ_map(shotnumber, diag='EQH', exp='AUGD', 
                          ed=0)
    except:
        eqh = dd.shotfile(diagnostic='EQI', pulseNumber=shotnumber,
                      experiment=exp, edition=0)
        equ = meq.equ_map(shotnumber, diag='EQI', exp='AUGD', 
                          ed=0)
        
    # Getting the calibration.
    try:
        cal_shot = dd.getLastShotNumber(diagnostic=b'RMD', 
                                        pulseNumber=shotnumber,
                                        experiment=b'AUGD')
        
        sf_rmd = dd.shotfile(diagnostic='RMD', pulseNumber=cal_shot, 
                         experiment='AUGD', edition=0)
    except:
        raise Exception('Could not retrieve the ECE calibration')
        
    #--- Time base retrieval.
    try:
        # Getting the fastest time basis.
        time = sf_rmc(name='TIME-AD0')
    except:
        raise Exception('Time base not available in shotfile!')
        
        
    if timeWindow is None:
        timeWindow = [time[0], time[-1]]
        
    warnings.filterwarnings('ignore')
    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[-1])
    
    #--- Getting the calibration from RMD
    mult_cal = sf_rmd.getParameter('eCAL-A1', 'MULTIA00').data
    mult_cal = np.append(mult_cal, 
              sf_rmd.getParameter('eCAL-A2', 'MULTIA00').data)
    
    shift_cal = sf_rmd.getParameter('eCAL-A1', 'SHIFTB00').data
    shift_cal = np.append(shift_cal, 
              sf_rmd.getParameter('eCAL-A2', 'SHIFTB00').data)
    
    # Getting the availability.
    avail_cec = sf_cec.getParameter('parms-A', 'AVAILABL').data.astype(bool)
    avail_rmd = sf_rmd.getParameter('parms-A', 'AVAILABL').data.astype(bool)
    avail_rmc = sf_rmc.getParameter('parms-A', 'AVAILABL').data.astype(bool)
    avail_flag = np.array((avail_cec * avail_rmd * avail_rmc), dtype=bool)
    
    del avail_cec
    del avail_rmd
    del avail_rmc
    
    # Getting the positions (R, Z) -> rho_pol
    rA_rmd = sf_rmd(name='R-A', tBegin=timeWindow[0]-dt, 
                    tEnd=timeWindow[1]+dt)
    zA_rmd = sf_rmd(name='z-A', tBegin=timeWindow[0]-dt, 
                    tEnd=timeWindow[1]+dt)
    
    if fast:
        rA= np.mean(rA_rmd.data, axis=0)
        zA= np.mean(zA_rmd.data, axis=0)
        time_avg = np.mean(timeWindow)
        rhop = equ.rz2rho(rA, zA,t_in=time_avg, coord_out='rho_pol',
                          extrapolate=True).flatten()
        rhot = equ.rz2rho(rA, zA,t_in=time_avg, coord_out='rho_tor',
                          extrapolate=True).flatten()
    else:
        # As the R and z are not calculated for all the points that ECE is
        # measuring, we build here an interpolator.
        rA_interp = interp1d(rA_rmd.time, rA_rmd.data, axis = 0,
                             fill_value=np.nan, kind=method,
                             assume_sorted=True, bounds_error=False)
        zA_interp = interp1d(zA_rmd.time, zA_rmd.data, axis = 0,
                             fill_value=np.nan, kind=method,
                             assume_sorted=True, bounds_error=False)
        
        # Getting the rho_pol equivalent to the positions.
        
        rhop = np.zeros(rA_rmd.data.shape)
        rhot = np.zeros(rA_rmd.data.shape)
        for ii in tqdm(np.arange(rA_rmd.data.shape[0])):
            rhop[ii, :] = equ.rz2rho(rA_rmd.data[ii, :], zA_rmd.data[ii, :],
                                     t_in=rA_rmd.time[ii], 
                                     coord_out='rho_pol',
                                     extrapolate=True)
            rhot[ii, :] = equ.rz2rho(rA_rmd.data[ii, :], zA_rmd.data[ii, :],
                                      t_in=rA_rmd.time[ii], 
                                      coord_out='rho_tor',
                                      extrapolate=True)
        
            
        # Creating the interpolators.
        rhop_interp = interp1d(rA_rmd.time, rhop, axis = 0,
                               fill_value=np.nan, kind=method,
                               assume_sorted=True, bounds_error=False)
        rhot_interp = interp1d(rA_rmd.time, rhot, axis = 0,
                               fill_value=np.nan, kind=method,
                               assume_sorted=True, bounds_error=False)
    equ.Close()
    sf_rmd.close()
    
    
    #--- Getting the electron temperature from the RMC shotfile.
    Trad_rmc1 = sf_rmc(name='Trad-A1', 
                       tBegin=timeWindow[0], tEnd=timeWindow[1])
    Trad_rmc2 = sf_rmc(name='Trad-A2', 
                       tBegin=timeWindow[0], tEnd=timeWindow[1])
    
    sf_rmc.close()
    time = Trad_rmc1.time
    Trad_rmc  = np.append(Trad_rmc1.data, Trad_rmc2.data, axis = 1)
    
    # With the new time-basis, we compute the actual positions of the ECE
    # radiometers in time.
    if fast:
        Rece = rA
        Zece = zA
        RhoP_ece = rhop
        RhoT_ece = rhot
    else:
        Rece = rA_interp(time)
        Zece = zA_interp(time)
        RhoP_ece = rhop_interp(time)
        RhoT_ece = rhot_interp(time)
        del rA_interp
        del zA_interp
        del rhop_interp
        del rhot_interp
        
    
    del Trad_rmc1 # Releasing some memory.
    del Trad_rmc2 # Releasing some memory.
    del rA_rmd
    del zA_rmd
    del rhop
    del rhot
    
    #--- Applying the calibration.
    Trad = np.zeros(Trad_rmc.shape)      # Electron temp [eV]
    Trad_norm = np.zeros(Trad_rmc.shape) # Norm'd by the maximum.
    for ii in np.arange(Trad_rmc.shape[1]):
        Trad[:, ii] = Trad_rmc[:, ii]*mult_cal[ii] + shift_cal[ii]
        meanTe = np.mean(Trad[:, ii])
        Trad_norm[:, ii] =  Trad[:, ii]/meanTe
    
    del Trad_rmc # Release the uncalibrated signal from memory.
    
    #--- Checking that the slow channel is consistent with the fast-channels.
    Trad_slow = sf_cec('Trad-A', tBegin = timeWindow[0], tEnd = timeWindow[1])
    sf_cec.close()
    meanECE   = np.mean(Trad_slow.data, axis=0)
    maxRMD    = np.max(Trad, axis=0)
    minRMD    = np.min(Trad, axis=0)
    RmeanRMD = np.mean(Rece.flatten())
    
    # We erase the channels that have and average in the slow channels that
    # are not within the RMD signals. Also, take away those not within the
    # vessel.
    avail_flag &= ((meanECE > minRMD) & (meanECE < maxRMD)) & (RmeanRMD > 1.03)
    del meanECE
    del maxRMD
    del minRMD
    
    #--- Applying the rhop limits.
    if rhopLimits is None:
        rhopLimits = [-1.0, 1.0]
        
    if fast:
        avail_flag &= ((RhoP_ece > rhopLimits[0]) & \
                       (RhoP_ece < rhopLimits[1]))
        
        Rece = Rece[avail_flag]
        Zece = Zece[avail_flag]
        RhoP_ece = RhoP_ece[avail_flag]
        RhoT_ece = RhoT_ece[avail_flag]
        Trad = Trad[:, avail_flag]
        Trad_norm = Trad_norm[:, avail_flag]
    else: 
        flags = ((RhoP_ece > rhopLimits[0]) & (RhoP_ece < rhopLimits[1]))
        avail_flag &= np.all(flags, axis=0) # If the channel is continuously 
                                            # outside, then discard it.
        Rece = Rece[:, avail_flag] 
        Zece = Zece[:, avail_flag] 
        RhoP_ece = RhoP_ece[:, avail_flag] 
        RhoT_ece = RhoT_ece[:, avail_flag] 
        Trad = Trad[:, avail_flag]
        Trad_norm = Trad_norm[:, avail_flag]
        
        # Now, set to NaN the values that are outside the limits.
        flags = not flags[:, avail_flag]
        Rece[flags] = np.nan
        Zece[flags] = np.nan
        RhoP_ece[flags] = np.nan
        RhoT_ece[flags] = np.nan
        Trad[flags] = np.nan
        Trad_norm[flags] = np.nan
        
        
    #--- Adding a conventional sign to the rhopol and rhotor.
    # Getting the axis center.
    
    eqh_time = np.asarray(eqh(name='time').data)
    eqh_ssqnames = eqh.GetSignal(name='SSQnam')
    eqh_ssq = eqh.GetSignal(name='SSQ')
    
    t0 = (np.abs(eqh_time - timeWindow[0])).argmin() - 1
    t1 = (np.abs(eqh_time - timeWindow[1])).argmin() + 1
    
    ssq = dict()
    for jssq in range(eqh_ssq.shape[1]):
        tmp = b''.join(eqh_ssqnames[jssq,:]).strip()
        lbl = tmp.decode('utf8')
        if lbl.strip() != '':
            ssq[lbl] = eqh_ssq[t0:t1, jssq]
            
    if 'Rmag' not in ssq:
        print('Warning: Proceed with care, magnetic axis radius not found.\n \
              Using the geometrical axis.')
        if fast: 
            Raxis = 1.65
        else:
            Raxis = 1.65*np.ones(Rece.shape[0]) # [m]
    else:
        Raxis_eqh = ssq['Rmag']
        
        if fast:
            Raxis = np.mean(Raxis_eqh)
            sign_array = Rece < Raxis
            RhoP_ece[sign_array] = - RhoP_ece[sign_array]
        else: 
            eqh_time = eqh_time[t0:t1]
            Raxis = interp1d(eqh_time, Raxis_eqh, 
                             fill_value='extrapolate', kind=method,
                             assume_sorted=True, bounds_error=False)(time)
            
            # If R_ECE < R(magnetic axis), then the rhopol is negative.
            for ii in np.arange(Trad.shape[1]):
                sign_array = Rece[:,ii] < Raxis
                RhoP_ece[sign_array, ii] = - RhoP_ece[sign_array, ii]
    
    eqh.close()
    warnings.filterwarnings('default')
    
    #--- Saving output.
    output = {'time': np.asarray(time, dtype=np.float32),
              'r': np.asarray(Rece,dtype=np.float32),
              'z':np.asarray(Zece,dtype=np.float32),
              'rhop': np.asarray(RhoP_ece,dtype=np.float32),
              'rhot': np.asarray(RhoT_ece,dtype=np.float32),
              'Trad': np.asarray(Trad,dtype=np.float32),
              'Trad_norm': np.asarray(Trad_norm,dtype=np.float32),
              'Raxis': np.asarray(Raxis,dtype=np.float32),
              'fast_rhop': fast, 
              'active': avail_flag,
              'channels': (np.arange(len(avail_flag))+1)[avail_flag],
              'shotnumber': shotnumber
              }
    
    return output


def correctShineThroughECE(ecedata: dict, diag: str = 'PED', exp: str = 'AUGD',
                           edition: int = 0):
    """
    For a given data-set of the ECE data, a new entry will be provided with
    the temperature divided by the electron temperature gradient, trying to 
    correct the shine-through effect.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param ecedata: dictionary with the data from the ECE.
    @param diag: Diagnostic from which the eletron gradient will be retrieved.
    By default is set to PED in AUGD, because its separatrix value are more
    reliable. If the PED shotfile is not found, then the IDA shotfile will be
    opened.
    @parad exp: experiment from which the electron temperature will be 
    extracted. By default, AUGD.
    @param ed: Edition to retrieve from equilibrium. The latest is taken by
    default.
    """
    
    #--- Trying to open the PED shotfile.
    using_PED = False
    if diag == 'PED':
        sf = dd.shotfile(diagnostic='PED', experiment=exp,
                         pulseNumber=ecedata['shotnumber'], 
                         edition=edition)
        
        name_te = 'TeFit'
        name_rhop = 'rhoFit'
        using_PED = True
    else:
        sf = dd.shotfile(diagnostic='IDA', experiment='AUGD',
                         pulseNumber=ecedata['shotnumber'],
                         edition=edition)
        
        name_te = 'dTe_dr'
        name_rhop = 'rhop'
    
    rhop = sf(name=name_rhop, tBegin=ecedata['fft']['time'][0],
              tEnd=ecedata['fft']['time'][-1])
    
    te_data = sf(name=name_te, tBegin=ecedata['fft']['time'][0],
                 tEnd=ecedata['fft']['time'][-1])
    
    if using_PED:
        dte     = np.abs(np.diff(te_data.data.squeeze()))
        drhop   = np.diff(rhop.data.squeeze())
        dtedrho = dte/drhop
        rhop_c  = (rhop.data.squeeze()[1:] + rhop.data.squeeze()[:-1])/2.0
        
        dte_eval = interp1d(rhop_c, dtedrho, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')\
                            (np.abs(ecedata['rhop']))
                            
        ecedata['fft']['dTe'] = dtedrho
        ecedata['fft']['dTe_base'] = rhop_c
                            
        ecedata['fft']['spec_dte'] = ecedata['fft']['spec']   
        for ii in np.arange(ecedata['fft']['spec_dte'].shape[0]):
            for jj in np.arange(ecedata['fft']['spec_dte'].shape[1]):
                ecedata['fft']['spec_dte'][ii, jj, :] /= dte_eval
        
    else:
        time = sf(name='time')
        t0 = np.maximum(0, 
                        np.abs(time.flatten() - \
                               ecedata['fft']['time'][0]).argmin() - 1)
        t1 = np.minimum(len(time)-1, 
                        np.abs(time.flatten() - \
                               ecedata['fft']['time'][-1]).argmin() + 1)
        
        t0 = np.asarray(t0, dtype=int)
        t1 = np.asarray(t1, dtype=int)
        
        # Reducing the size, to make easier the interpolation.
        
        time = time[t0:t1]
        te_data = te_data.data[t0:t1, :]
        rhop = rhop[t0, :]
        
        dte_eval_fun = interp2d(time, rhop, np.abs(te_data.T))
        
        dte_eval = np.abs(dte_eval_fun(ecedata['fft']['time'], 
                                       np.abs((ecedata['rhop']))).T)
        
        ecedata['fft']['dTe_base'] = rhop
        ecedata['fft']['dTe'] = np.abs(np.mean(te_data, axis=0))
        ecedata['fft']['spec_dte'] = ecedata['fft']['spec']          
        for ii in np.arange(ecedata['Trad'].shape[1]):
            ecedata['fft']['spec_dte'][:, ii, :] /= dte_eval
    return ecedata
    
# -----------------------------------------------------------------------------
# --- Vessel coordinates
# -----------------------------------------------------------------------------
def poloidal_vessel(shot: int = 30585, simplified: bool = False):
    """
    Get coordinate of the poloidal projection of the vessel

    Jose Rueda: jrrueda@us.es

    @param shot: shot number to be used
    @param simplified: if true, a 'basic' shape of the poloidal vessel will be
    loaded, ideal for generate a 3D revolution surface from it
    """
    if simplified is not True:
        r = []
        z = []
        # Get vessel coordinates
        gc_r, gc_z = get_gc.get_gc(shot)
        for key in gc_r.keys():
            # print(key)
            r += list(gc_r[key][:])
            r.append(np.nan)
            z += list(gc_z[key][:])
            z.append(np.nan)
        return np.array((r, z)).transpose()
    else:
        file = os.path.join(pa.ScintSuite, 'Data', 'Vessel', 'AUG_pol.txt')
        return np.loadtxt(file, skiprows=4)


def toroidal_vessel(rot: float = -np.pi/8.0*3.0):
    """
    Return the coordinates of the AUG vessel

    Jose Rueda Rueda: ruejo@ipp.mpg.de

    Note: x = NaN indicate the separation between vessel block

    @param rot: angle to rotate the coordinate system
    @return xy: np.array with the coordinates of the points [npoints, 2]
    """
    # --- Section 0: Read the data
    # The files are a series of 'blocks' representing each piece of the vessel,
    # each block is separated by an empty line. I will scan the file line by
    # line looking for the position of those empty lines:
    file = os.path.join(pa.ScintSuite, 'Data', 'Vessel', 'AUG_tor.txt')
    cc = 0
    nmax = 2000
    xy_vessel = np.zeros((nmax, 2))
    with open(file) as f:
        # read the comment block
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        # read the vessel components:
        for i in range(nmax):
            line = f.readline()
            if line == '\n':
                xy_vessel[cc, 0] = np.nan
            elif line == '':
                break
            else:
                dummy = line.split()
                xx = np.float(dummy[0])
                yy = np.float(dummy[1])
                xy_vessel[cc, 0] = xx * np.cos(rot) - yy * np.sin(rot)
                xy_vessel[cc, 1] = xx * np.sin(rot) + yy * np.cos(rot)
            cc += 1
    return xy_vessel[:cc-1, :]

# -----------------------------------------------------------------------------
# --- Magnetics
# -----------------------------------------------------------------------------
mag_coils_grp2coilName = { 'C07': ['C07', np.arange(1, 32)],
                           'C09': ['C09', np.arange(1, 32)],
                           'B-31_5_11': ['B31', np.arange(5, 11)],
                           'B-31_32_27' :['B31', np.arange(32, 38)]
                            }


def get_magnetics(shotnumber: int, coilNumber: int, coilGroup: str = 'B31',
                  timeWindow: float = None):
    """
    Retrieves from the shot file the magnetic data information.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param shotnumber: Shot number to get the data.
    @param coilNumber: Coil number in the coil array.
    @param coilGroup: can be B31, B17, C09,... by default set to B31
    (ballooning coils)
    @param timeWindow: Time window to get the magnetic data. If None, all the
    time window will be obtained.
    @return output: magnetic data (time traces and position.)
    """
    
    if shotnumber <= 33739:
        diag = 'MHA'
    else:
        diag = 'MHI'
        
    exp = 'AUGD' # Only the AUGD data retrieve is supported with the dd.
    
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp,  edition=0)
    except:
        raise Exception('Shotfile not existent for '+diag+' #'+str(shotnumber))
    
    
    try:
        # Getting the time base.
        time = sf(name='Time')
    except:
        raise Exception('Time base not available in shotfile!')
        
        
    if timeWindow is None:
        timeWindow = [time[0], time[-1]]
    
    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[1])
    
    name = '%s-%02d'%(coilGroup, coilNumber)
    try:
        # Getting the time base.
        mhi = sf(name=name, tBegin=timeWindow[0], tEnd=timeWindow[1])
    except:
        raise Exception(name+' not available in shotfile.')
        sf.close()
    
    sf.close()
    #--- Getting the calibration factors from the CMH shotfile.
    # Get the last shotnumber where the calibration is written.
    cal_shot = dd.getLastShotNumber(diagnostic=b'CMH', pulseNumber=shotnumber,
                                    experiment=b'AUGD')
    
    sf = dd.shotfile(diagnostic='CMH', pulseNumber=cal_shot, 
                     experiment='AUGD', edition=0)
    cal_name = 'C'+name
    cal = {'R': sf.getParameter(setName=cal_name, parName=b'R').data,
           'z': sf.getParameter(setName=cal_name, parName=b'z').data,
           'phi': sf.getParameter(setName=cal_name, parName=b'phi').data,
           'theta': sf.getParameter(setName=cal_name, parName=b'theta').data,
           'EffArea': sf.getParameter(setName=cal_name, 
                                      parName=b'EffArea').data
          }
        
        
    t0 = np.abs(mhi.time-timeWindow[0]).argmin()
    t1 = np.abs(mhi.time-timeWindow[-1]).argmin()
    output = {'time': mhi.time[t0:t1],
              'data': mhi.data[t0:t1],
              'R': cal['R'], 
              'z': cal['z'], 
              'phi': cal['phi'],
              'theta':cal['theta'],
              'area': cal['EffArea']
             }
    
    sf.close()
    
    return output


def get_magnetic_poloidal_grp(shotnumber: int, timeWindow: float, 
                              coilGrp: int=None):
    
    """
    It retrieves from the database the data of a set of magnetic coils lying
    within the same phi-angle. The way to choose them is either providing the
    coils group as provided above or giving the phi angle, and the nearest set
    of coils will be obtained. For example:
        phi_range = 45º will provide the C07 coil group.
        
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param shotnumber: Shotnumber to retrieve the magnetic coils.
    @param timeWindow: time window to get the data.
    @param coilGrp: 'C07', 'C09', 'B31-5_11', 'B31-32-38' depending upon the
    group that needs to be read from database.
    @param phi_mag: each coil group is characterized by an approximate phi  
    angle. If this is provided, the nearest group in phi is retrieved from the 
    database. See variable @see{MHI_GROUP_APPR_PHI} to see which are the 
    angles used.
    
    @return output: group of the magnetic coils.
    """
    # --- Parsing the inputs
    if coilGrp is None:
        raise Exception('Either the coil-group or the angle must be provided.')
    
    # If the time window is a single float, we will make an array containing 
    # at least a 5ms window.
    dt = 5e-3 
    if len(timeWindow) == 1:
        timeWindow = np.array((timeWindow, timeWindow + dt))
        
    # --- Parsing the magnetic group input
    # Check if it is in the list of coils.
    if coilGrp not in mag_coils_grp2coilName:
        raise Exception('Coil group, %s, not found!'%coilGrp)
        
    # Getting the list with the coils names.
    coil_list = mag_coils_grp2coilName[coilGrp]
    numcoils = len(coil_list[1])
    
    # --- Opening the shotfile.
    if shotnumber <= 33739:
        diag = 'MHA'
    else:
        diag = 'MHI'
        
    exp = 'AUGD' # Only the AUGD data retrieve is supported with the dd.
    
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp,  edition=0)
    except:
        raise Exception('Shotfile not existent for '+diag+' #'+str(shotnumber))
    
    
    try:
        # Getting the time base.
        time = sf(name='Time')
    except:
        raise Exception('Time base not available in shotfile!')
        
        
    if timeWindow is None:
        timeWindow = [time[0], time[-1]]
    
    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[1])
    
    
    #--- Getting the calibration factors from the CMH shotfile.
    # Get the last shotnumber where the calibration is written.
    try:
        cal_shot = dd.getLastShotNumber(diagnostic=b'CMH', pulseNumber=shotnumber,
                                        experiment=b'AUGD')
        cal_sf = dd.shotfile(diagnostic='CMH', pulseNumber=cal_shot, 
                             experiment='AUGD', edition=0)
    except:
        sf.close()
        raise Exception('Could not get the calibration data.')
    
    # --- Getting the coils data.    
    output = { 'phi': np.zeros((numcoils,)),
               'theta': np.zeros((numcoils,)),
               'dtheta': np.zeros((numcoils,)),
               'R': np.zeros((numcoils,)),
               'z': np.zeros((numcoils,)),
               'area': np.zeros((numcoils,)),
               'time': [], 
               'data': [],
               'coilNumber': np.zeros((numcoils,)),
             }
    
    jj = 0
    flags = np.zeros((numcoils,), dtype=bool)
    for ii in tqdm(np.arange(numcoils)):
        name = '%s-%02d'%(coil_list[0], coil_list[1][ii])
        cal_name = 'C'+name
        
        
        try:
            # Try to get the magnetic data.
            mhi = sf(name=name, tBegin=timeWindow[0], 
                                tEnd=timeWindow[-1])
            
            # Try to get the calibration.
            cal = {'R': cal_sf.getParameter(setName=cal_name, 
                                            parName=b'R').data,
                   
                   'z': cal_sf.getParameter(setName=cal_name,
                                            parName=b'z').data,
                   
                   'phi': cal_sf.getParameter(setName=cal_name, 
                                          parName=b'phi').data,
                   
                   'theta': cal_sf.getParameter(setName=cal_name, 
                                            parName=b'theta').data,
                   
                   'dtheta': cal_sf.getParameter(setName=cal_name, 
                                                 parName=b'dtheta').data,
                   
                   'area': cal_sf.getParameter(setName=cal_name, 
                                          parName=b'EffArea').data
                       } 
        except:
            print('Could not retrieve coils %s-%02d'%
                 (coil_list[0],coil_list[1][ii]))
            continue
        
        
        t0 = np.abs(mhi.time-timeWindow[0]).argmin()  # Beginning time index.
        t1 = np.abs(mhi.time-timeWindow[-1]).argmin() # Ending time index.
        
        if jj == 0:
            output['time'] = mhi.time[t0:t1]
            output['data'] = mhi.data[t0:t1]
        else:
            output['data'] = np.vstack((output['data'], mhi.data[t0:t1]))
        
        output['phi'][ii] = cal['phi']
        output['theta'][ii] = cal['theta']
        output['dtheta'][ii] = cal['dtheta']
        output['R'][ii] = cal['R']
        output['z'][ii] = cal['z']
        output['area'][ii] = cal['area']
        output['coilNumber'][ii] = ii+1
        flags[ii] = True
        
        jj += 1
        
        del mhi
        del cal
        
    #--- All the coils have been read.
    # Removing the holes left by the coils that were not available in MHI.
    output['phi']  = output['phi'][flags]
    output['theta']  = output['theta'][flags]
    output['dtheta']  = output['dtheta'][flags]
    output['R']  = output['R'][flags]
    output['z']  = output['z'][flags]
    output['area']  = output['area'][flags]
    output['coilNumber']  = output['coilNumber'][flags]
    # --- Closing shotfiles.
    sf.close()
    cal_sf.close()
        
    return output


# -----------------------------------------------------------------------------
# --- NBI coordinates
# -----------------------------------------------------------------------------
def _NBI_diaggeom_coordinates(nnbi):
    """
    Just the coordinates manually extracted for shot 32312

    @param nnbi: the NBI number
    @return coords: dictionary containing the coordinates of the initial and
    final points. '0' are near the source, '1' are near the central column
    """
    r0 = np.array([2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6])
    r1 = np.array([1.046, 1.046, 1.046, 1.046, 1.048, 2.04, 2.04, 1.048])

    z0 = np.array([0.022, 0.021, -0.021, -0.022,
                   -0.019, -0.149, 0.149, 0.19])
    z1 = np.array([-0.12, -0.145, 0.145, 0.12, -0.180, -0.6, 0.6, 0.180])

    phi0 = np.array([-32.725, -31.88, -31.88, -32.725,
                     145.58, 148.21, 148.21, 145.58]) * np.pi / 180.0
    phi1 = np.array([-13.81, 10.07, 10.07, -13.81,
                     -180.0, -99.43, -99.43, -180.0]) * np.pi / 180.0

    x0 = r0 * np.cos(phi0)
    x1 = r1 * np.cos(phi1)

    y0 = r0 * np.sin(phi0)
    y1 = r1 * np.sin(phi1)

    coords = {'phi0': phi0[nnbi-1], 'phi1': phi1[nnbi-1],
              'x0': x0[nnbi-1], 'y0': y0[nnbi-1],
              'z0': z0[nnbi-1], 'x1': x1[nnbi-1],
              'y1': y1[nnbi-1], 'z1': z1[nnbi-1]}
    return coords

def getNBIwindow(timeWindow: float, shotnumber: int,
                 nbion: int, nbioff: int = None, 
                 simul: bool = True, pthreshold: float = 2.0):
    """
    Gets the time window within the limits provide within the timeWindow that
    corresponds to the list nbiON that are turned on and the list nbioff.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param timeWindow: window of time to retrieve the NBI data.
    @param shotnumber: Shot number from where to take the NBI timetraces.
    @param nbion: list with the NBI number that should be ON.
    @param nbioff: list with the NBIs that should be OFF.
    @param simul: simultaneous flag. If True all the NBIs of nbion should be
    ON simultaenously.
    @param pthreshold: power threshold to consider the beam is ON [MW]. 
    Default to 2.0 MW (to choose the 2.5MW standard beam.)
    """
    
    # --- Checking the time inputs.
    if len(timeWindow) == 1:
        timeWindow = np.array((timeWindow, np.inf))
        
    elif np.mod(len(timeWindow), 2) != 0:
        timeWindow[len(timeWindow)] = np.inf
        
    # --- Opening the NBIs shotfile.
    try:
        sf = dd.shotfile(diagnostic='NIS', pulseNumber=shotnumber,
                         experiment='AUGD', edition=0)
    except:
        raise Exception('Could not open NIS shotfile for #$05d'%shotnumber)

    # --- Transforming the indices of the NBIs into the AUG system (BOX, Beam)
    nbion_box = np.asarray(np.floor(nbion/4), dtype=int)
    nbion_idx = np.asarray(nbion - (nbion_box+1)*4 - 1, dtype=int)
    if nbioff is not None:
        nbioff_box = np.asarray(np.floor(nbioff/4), dtype=int)
        nbioff_idx = np.asarray(nbioff - (nbioff_box+1)*4 - 1, dtype=int)
    
    #--- Reading the NBI data.
    pniq = np.transpose(sf.getObjectData(b'PNIQ'), (2, 0, 1))*1.0e-6
    timebase = sf.getTimeBase(b'PNIQ')
    sf.close()
    
    t0_0 = np.abs(timebase-timeWindow[0]).argmin()
    t1_0 = np.abs(timebase-timeWindow[-1]).argmin()
    # Selecting the NBIs.
    pniq_on  = pniq[t0_0:t1_0, nbion_box,  nbion_idx]  > pthreshold
    if nbioff is not None:
        pniq_off = pniq[t0_0:t1_0, nbioff_box, nbioff_idx] > pthreshold
    timebase = timebase[t0_0:t1_0]
    
    # --- Reshaping the PNIQ into a 2D matrix for easier handling.
    
    if len(nbion) == 1:
        pniq_on = np.reshape(pniq_on, (len(pniq_on),1))
    else:
        pniq_on = pniq_on.reshape((pniq_on.shape[0], 
                                   pniq_on.shape[1]*pniq_on.shape[2]))
     
    if nbioff is not None:
        if len(nbioff) == 1:
            pniq_off = np.reshape(pniq_off, (len(pniq_off),1))
        else:
            pniq_off = pniq_off.reshape((pniq_off.shape[0], 
                                         pniq_off.shape[1]*pniq_off.shape[2]))
    
    if simul: # if all the NBIs must be simultaneously ON.
        auxON = np.all(pniq_on, axis=1)
    else: # if only one of the beams must be turned on.
        auxON = np.any(pniq_on, axis=1)
    
    # We take out the times at which the NBI_OFF are ON.
    if nbioff is not None:
        auxOFF = np.logical_not(np.any(pniq_off, axis=1))
        
        # Making the AND operation for all the times.
        aux = np.logical_and(auxON, auxOFF)
    else:
        aux = auxON
   
    # --- Loop over the time windows.
    nwindows = np.floor(len(timeWindow)/2)
    flags = np.zeros((pniq_on.shape[0],), dtype=bool)
    for ii in np.arange(nwindows, dtype=int):
        t0 = np.abs(timebase-timeWindow[2*ii]).argmin()
        t1 = np.abs(timebase-timeWindow[2*ii + 1]).argmin()
        
        flags[t0:t1] = True
    
            
    # --- Filtering the outputs.
    aux = np.logical_and(flags, aux)
    data = pniq[t0_0:t1_0, nbion_box,  nbion_idx]
    output = { 'timewindow': timeWindow,
               'flags': aux,
               'time': timebase[aux],
               'data': data[aux, ...]
             }
    return output

# -----------------------------------------------------------------------------
# --- Other shot files
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, channels, shot: int):
    """
    Get the signal for the fast channels (PMT, APD)

    Jose Rueda Rueda: jrrueda@us.es

    @param diag: diagnostic: 'FILD' or 'INPA'
    @param diag_number: 1-5
    @param channels: channel number we want, or arry with channels
    @param shot: shot file to be opened
    """
    # Check inputs:
    if not ((diag == 'FILD') or (diag != 'INPA')):
        raise Exception('No understood diagnostic')

    # Load diagnostic names:
    if diag == 'FILD':
        if (diag_number > 5) or (diag_number < 1):
            print('You requested: ', diag_number)
            raise Exception('Wrong fild number')
        diag_name = fild_diag[diag_number - 1]
        signal_prefix = fild_signals[diag_number - 1]
        nch = fild_number_of_channels[diag_number - 1]

    # Look which channels we need to load:
    try:    # If we received a numpy array, all is fine
        nch_to_load = channels.size
        ch = channels
    except AttributeError:  # If not, we need to create it
        ch = np.array([channels])
        nch_to_load = ch.size

    # Open the shot file
    fast = dd.shotfile(diag_name, shot)
    data = []
    for ic in range(nch):
        real_channel = ic + 1
        if real_channel in ch:
            name_channel = signal_prefix + "{0:02}".format(real_channel)
            channel_dat = fast.getObjectData(name_channel.encode('UTF-8'))
            data.append(channel_dat)
        else:
            data.append(None)
    # get the time base (we will use last loaded channel)
    time = fast.getTimeBase(name_channel.encode('UTF-8'))
    print('Number of requested channels: ', nch_to_load)
    return {'time': time, 'signal': data}


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class NBI:
    """Class with the information and data from an NBI"""

    def __init__(self, nnbi: int, shot: int = 32312, diaggeom=True):
        """
        Initialize the class

        @todo: Implement the actual algorithm to look at the shotfiles for the
        NBI geometry
        @todo: Create a new package to set this structure as machine
        independent??

        @param    nnbi: number of the NBI
        @param    shot: shot number
        @param    diaggeom: If true, values extracted manually from diaggeom
        """
        ## NBI number:
        self.number = nnbi
        ## Coordinates of the NBI
        self.coords = None
        ## Pitch information (injection pitch in each radial position)
        self.pitch_profile = None
        if diaggeom:
            self.coords = _NBI_diaggeom_coordinates(nnbi)
        else:
            raise Exception('Sorry, option not jet implemented')

    def calc_pitch_profile(self, shot: int, time: float, rmin: float = 1.4,
                           rmax: float = 2.2, delta: float = 0.04,
                           BtIp: int = -1.0, deg: bool = False):
        """
        Calculate the pitch profile of the NBI along the injection line

        If the 'pitch_profile' field of the NBI object is not created, it
        initialize it, else, it just append the new time point (it will insert
        the time point at the end, if first you call the function for t=1.0,
        then for 0.5 and then for 2.5 you will create a bit of a mesh, please
        use this function with a bit of logic)

        DISCLAIMER: We will not check if the radial position coincides with the
        previous data in the pitch profile structure, it is your responsibility
        to use consistent input when calling this function

        @todo implement using insert the insertion of the data on the right
        temporal position

        @param shot: Shot number
        @param time: Time in seconds
        @param rmin: miminum radius to be considered during the calculation
        @param rmax: maximum radius to be considered during the calculation
        @param delta: the spacing of the points along the NBI [m]
        @param BtIp: sign of the magnetic field respect to the current, the
        pitch will be defined as BtIp * v_par / v
        @param deg: If true the pitch is acos(BtIp * v_par / v)
        """
        if self.coords is None:
            raise Exception('Sorry, NBI coordinates are needed!!!')
        # Get coordinate vector
        v = np.array([self.coords['x1'] - self.coords['x0'],
                      self.coords['y1'] - self.coords['y0'],
                      self.coords['z1'] - self.coords['z0']])
        normv = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        # make the vector with the desired length
        v *= delta / normv
        # estimate the number of steps
        nstep = np.int(normv / delta) + 10
        # 'walk' along the NBI
        point = np.array([self.coords['x0'], self.coords['y0'],
                          self.coords['z0']])
        R = np.zeros(nstep)
        Z = np.zeros(nstep)
        phi = np.zeros(nstep)
        flags = np.zeros(nstep, dtype=np.bool)
        for istep in range(nstep):
            Rdum = np.sqrt(point[0]**2 + point[1]**2)
            if Rdum < rmin:
                break
            if Rdum < rmax:
                R[istep] = Rdum
                Z[istep] = point[2]
                phi[istep] = np.arctan2(point[1], point[0])
                flags[istep] = True
            point = point + v
        # calculate the magnetic field
        R = R[flags]
        Z = Z[flags]
        phi = phi[flags]
        ngood = R.size
        pitch = np.zeros(ngood)
        br, bz, bt, bp = get_mag_field(shot, R, Z, time=time)
        bx = -np.cos(0.5*np.pi - phi) * bt + np.cos(phi) * br
        by = np.sin(0.5*np.pi - phi) * bt + np.sin(phi) * br
        B = np.vstack((bx, by, bz))
        bnorm = np.sqrt(np.sum(B**2, axis=0))
        pitch = (bx * v[0] + by * v[1] + bz * v[2]) / delta / bnorm
        pitch = BtIp * pitch.squeeze()
        if deg:
            pitch = np.arccos(pitch) * 180.0 / np.pi
        # Now we have the pitch profiles, we just need to store the info at the
        # right place
        if self.pitch_profile is None:
            self.pitch_profile = {'t': np.array(time),
                                  'z': Z, 'R': R, 'pitch': pitch}

        else:
            # number of already present times:
            nt = len(self.pitch_profile['t'])
            # see if the number of points along the NBI matches
            npoints = self.pitch_profile['R'].size
            if npoints / nt != R.size:
                raise Exception('Have you changed delta from the last run?')
            # insert the date where it should be
            self.pitch_profile['t'] = \
                np.vstack((self.pitch_profile['t'], time))
            self.pitch_profile['z'] = \
                np.vstack((self.pitch_profile['z'], Z))
            self.pitch_profile['R'] = \
                np.vstack((self.pitch_profile['R'], R))
            self.pitch_profile['pitch'] = \
                np.vstack((self.pitch_profile['pitch'], pitch))

    def plot_pitch_profile(self, line_param: dict = {'linewidth': 2},
                           ax_param={'grid': 'both', 'xlabel': 'R [cm]',
                                     'ylabel': '$\\lambda$', 'fontsize': 14},
                           ax=None):
        """
        Plot the NBI pitch profile

        Jose Rueda: jrrueda@us.es

        @param line_param: Dictionary with the line params
        @param ax_param: Dictionary with the param fr ax_beauty
        @param ax: axis where to plot, if none, open new figure
        @return : Nothing
        """
        if self.pitch_profile is None:
            raise Exception('You must calculate first the pitch profile')
        if ax is None:
            fig, ax = plt.subplots()
            ax_created = True
        ax.plot(self.pitch_profile['R'], self.pitch_profile['pitch'],
                **line_param, label='NBI#'+str(self.number))

        if ax_created:
            ax = ssplt.axis_beauty(ax, ax_param)
        try:
            plt.legend(fontsize=ax_param['fontsize'])
        except KeyError:
            print('You did not set the fontsize in the input params...')
