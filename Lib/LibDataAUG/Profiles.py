"""Routines to interact with the AUG database"""

import dd                # Module to load shotfiles
import numpy as np
import map_equ as meq    # Module to map the equilibrium
import warnings
import DiagParam as params
from tqdm import tqdm
from scipy.interpolate import interpn, interp1d, interp2d
from LibPaths import Path
pa = Path()


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

    # --- Opening the shotfiles.
    exp = 'AUGD'  # Only the AUGD data retrieve is supported with the dd.
    dt = 2*5.0e-3  # To get a bit more on the edges to make a better interp.
    method = 'linear'  # Interpolation method
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

    # --- Time base retrieval.
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

    # --- Getting the calibration from RMD
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
        rA = np.mean(rA_rmd.data, axis=0)
        zA = np.mean(zA_rmd.data, axis=0)
        time_avg = np.mean(timeWindow)
        rhop = equ.rz2rho(rA, zA, t_in=time_avg, coord_out='rho_pol',
                          extrapolate=True).flatten()
        rhot = equ.rz2rho(rA, zA, t_in=time_avg, coord_out='rho_tor',
                          extrapolate=True).flatten()
    else:
        # As the R and z are not calculated for all the points that ECE is
        # measuring, we build here an interpolator.
        rA_interp = interp1d(rA_rmd.time, rA_rmd.data, axis=0,
                             fill_value=np.nan, kind=method,
                             assume_sorted=True, bounds_error=False)
        zA_interp = interp1d(zA_rmd.time, zA_rmd.data, axis=0,
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
        rhop_interp = interp1d(rA_rmd.time, rhop, axis=0,
                               fill_value=np.nan, kind=method,
                               assume_sorted=True, bounds_error=False)
        rhot_interp = interp1d(rA_rmd.time, rhot, axis=0,
                               fill_value=np.nan, kind=method,
                               assume_sorted=True, bounds_error=False)
    equ.Close()
    sf_rmd.close()

    # --- Getting the electron temperature from the RMC shotfile.
    Trad_rmc1 = sf_rmc(name='Trad-A1',
                       tBegin=timeWindow[0], tEnd=timeWindow[1])
    Trad_rmc2 = sf_rmc(name='Trad-A2',
                       tBegin=timeWindow[0], tEnd=timeWindow[1])

    sf_rmc.close()
    time = Trad_rmc1.time
    Trad_rmc = np.append(Trad_rmc1.data, Trad_rmc2.data, axis=1)

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

    del Trad_rmc1  # Releasing some memory.
    del Trad_rmc2  # Releasing some memory.
    del rA_rmd
    del zA_rmd
    del rhop
    del rhot

    # --- Applying the calibration.
    Trad = np.zeros(Trad_rmc.shape)       # Electron temp [eV]
    Trad_norm = np.zeros(Trad_rmc.shape)  # Norm'd by the maximum.
    for ii in np.arange(Trad_rmc.shape[1]):
        Trad[:, ii] = Trad_rmc[:, ii]*mult_cal[ii] + shift_cal[ii]
        meanTe = np.mean(Trad[:, ii])
        Trad_norm[:, ii] = Trad[:, ii]/meanTe

    del Trad_rmc  # Release the uncalibrated signal from memory.

    # --- Checking that the slow channel is consistent with the fast-channels.
    Trad_slow = sf_cec('Trad-A', tBegin=timeWindow[0], tEnd=timeWindow[1])
    sf_cec.close()
    meanECE = np.mean(Trad_slow.data, axis=0)
    maxRMD = np.max(Trad, axis=0)
    minRMD = np.min(Trad, axis=0)
    RmeanRMD = np.mean(Rece.flatten())

    # We erase the channels that have and average in the slow channels that
    # are not within the RMD signals. Also, take away those not within the
    # vessel.
    avail_flag &= ((meanECE > minRMD) & (meanECE < maxRMD)) & (RmeanRMD > 1.03)
    del meanECE
    del maxRMD
    del minRMD

    # --- Applying the rhop limits.
    if rhopLimits is None:
        rhopLimits = [-1.0, 1.0]

    if fast:
        avail_flag &= ((RhoP_ece > rhopLimits[0]) &
                       (RhoP_ece < rhopLimits[1]))

        Rece = Rece[avail_flag]
        Zece = Zece[avail_flag]
        RhoP_ece = RhoP_ece[avail_flag]
        RhoT_ece = RhoT_ece[avail_flag]
        Trad = Trad[:, avail_flag]
        Trad_norm = Trad_norm[:, avail_flag]
    else:
        flags = ((RhoP_ece > rhopLimits[0]) & (RhoP_ece < rhopLimits[1]))
        avail_flag &= np.all(flags, axis=0)  # If the channel is continuously
        #                                    # outside, then discard it.
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

    # --- Adding a conventional sign to the rhopol and rhotor.
    # Getting the axis center.

    eqh_time = np.asarray(eqh(name='time').data)
    eqh_ssqnames = eqh.GetSignal(name='SSQnam')
    eqh_ssq = eqh.GetSignal(name='SSQ')

    t0 = (np.abs(eqh_time - timeWindow[0])).argmin() - 1
    t1 = (np.abs(eqh_time - timeWindow[1])).argmin() + 1

    ssq = dict()
    for jssq in range(eqh_ssq.shape[1]):
        tmp = b''.join(eqh_ssqnames[jssq, :]).strip()
        lbl = tmp.decode('utf8')
        if lbl.strip() != '':
            ssq[lbl] = eqh_ssq[t0:t1, jssq]

    if 'Rmag' not in ssq:
        print('Warning: Proceed with care, magnetic axis radius not found.\n \
              Using the geometrical axis.')
        if fast:
            Raxis = 1.65
        else:
            Raxis = 1.65 * np.ones(Rece.shape[0])  # [m]
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
                sign_array = Rece[:, ii] < Raxis
                RhoP_ece[sign_array, ii] = - RhoP_ece[sign_array, ii]

    eqh.close()
    warnings.filterwarnings('default')

    # --- Saving output.
    output = {
        'time': np.asarray(time, dtype=np.float32),
        'r': np.asarray(Rece, dtype=np.float32),
        'z': np.asarray(Zece, dtype=np.float32),
        'rhop': np.asarray(RhoP_ece, dtype=np.float32),
        'rhot': np.asarray(RhoT_ece, dtype=np.float32),
        'Trad': np.asarray(Trad, dtype=np.float32),
        'Trad_norm': np.asarray(Trad_norm, dtype=np.float32),
        'Raxis': np.asarray(Raxis, dtype=np.float32),
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
    # --- Trying to open the PED shotfile.
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
        dte = np.abs(np.diff(te_data.data.squeeze()))
        drhop = np.diff(rhop.data.squeeze())
        dtedrho = dte/drhop
        rhop_c = (rhop.data.squeeze()[1:] + rhop.data.squeeze()[:-1])/2.0

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
                        np.abs(time.flatten()
                               - ecedata['fft']['time'][0]).argmin() - 1)
        t1 = np.minimum(len(time)-1,
                        np.abs(time.flatten()
                               - ecedata['fft']['time'][-1]).argmin() + 1)

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
        diag_name = params.fild_diag[diag_number - 1]
        signal_prefix = params.fild_signals[diag_number - 1]
        nch = params.fild_number_of_channels[diag_number - 1]

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
