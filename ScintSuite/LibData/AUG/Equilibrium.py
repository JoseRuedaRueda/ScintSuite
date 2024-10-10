"""Routines for the magnetic equilibrium"""
import warnings
import numpy as np
import xarray as xr
import aug_sfutils as sf
import ScintSuite.errors as errors
from scipy.interpolate import interpn, interp1d

# --- Module hardcored parameters
ECRH_POWER_THRESHOLD = 0.05  # Threshold to consider ECRH on [MW]


# -----------------------------------------------------------------------------
# --- Magnetic field
# -----------------------------------------------------------------------------
def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, equ=None, **kwargs):
    """
    Wrapp to get AUG magnetic field

    Jose Rueda: jrrueda@us.es

    Note: No extra arguments are expected, **kwargs is just included for
    compatibility of the call to this method in other databases (machines)

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  diag: Diag for AUG database, default EQH
    :param  exp: experiment, default AUGD
    :param  ed: edition, default 0 (last)
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    :param  equ: equilibrium object from the library aug_sfutils

    :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T]
    
    @TODO: Include the sign of Bpol
    """
    # If the equilibrium object is not an input, let create it
    # created = False
    if equ is None:
        equ = sf.EQU(shot, diag=diag, ed=ed, exp=exp)
    # Now calculate the field
    # br, bz, bt = equ.rz2brzt(Rin, zin, t_in=time)
    br, bz, bt = sf.rz2brzt(equ, r_in=Rin, z_in=zin, t_in=time)
    bp = np.hypot(br, bz)

    return br, bz, bt, bp


# -----------------------------------------------------------------------------
# --- Flux coordinate
# -----------------------------------------------------------------------------
def get_mag_axis(shot, time: float = None, diag: str = 'GQH'):
    """
    Get the coordinates of the magnetic axis
    """
    sfo = sf.SFREAD(diag, shot)
    rmag = sfo('Rmag')
    zmag = sfo('Zmag')
    timebase = sfo.gettimebase('Rmag')
    if time is not None:
        rmag = interp1d(timebase, rmag)(time)
        zmag = interp1d(timebase, zmag)(time)
        timebase = time
    return rmag, zmag, time


def get_rho(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
            ed: int = 0, time: float = None, equ=None,
            coord_out: str = 'rho_pol'):
    """
    Wrapp to get AUG normalised radius.

    Jose Rueda: jrrueda@us.es

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  diag: Diag for AUG database, default EQH
    :param  exp: experiment, default AUGD
    :param  ed: edition, default 0 (last)
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    :param  equ: equilibrium object from the library map_equ
    :param  coord_out: the desired rho coordinate, default rho_pol

    :return rho: The desired rho coordinate evaluated at the points
    """
    # If the equilibrium object is not an input, let create it
    if equ is None:
        equ = sf.EQU(shot, diag=diag, exp=exp, ed=ed)
    # Now calculate the field
    rho = sf.rz2rho(equ, Rin, zin, t_in=time, coord_out=coord_out,
                    extrapolate=True)

    return rho


def get_rho2rz(shot: int, flxlabel: float, diag: str = 'EQH',
               exp: str = 'AUGD', ed: int = 0, time: float = None,
               coord_out: str = 'rho_pol', equ=None):
    """
    Gets the curves (R, z) associated to a given flux surface.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  flxlabel: flux surface label.
    :param  diag: Diag for AUG database, default EQH
    :param  exp: experiment, default AUGD
    :param  ed: edition, default 0 (last)
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    :param  equ: equilibrium object from the library map_equ
    :param  coord_out: the desired rho coordinate, default rho_pol
    """
    # If the equilibrium object is not an input, let create it
    if equ is None:
        equ = sf.EQU(shot, diag=diag, exp=exp, ed=ed)

    R, z = sf.rho2rz(equ, t_in=time, rho_in=flxlabel, coord_in=coord_out,
                     all_lines=False)

    if time is None:
        tout = equ.time
    else:
        tout = time

    return R, z, tout


def get_psipol(shot: int, Rin, zin, diag='EQH', exp: str = 'AUGD',
               ed: int = 0, time: float = None, equ=None):
    """
    Wrap to get AUG poloidal flux field

    Jose Rueda: jrrueda@us.es
    ft.
    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  diag: Diag for AUG database, default EQH
    :param  exp: experiment, default AUGD
    :param  ed: edition, default 0 (last)
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    :param  equ: equilibrium object from the library map_equ

    :return psipol: Poloidal flux evaluated in the input grid.
    """
    # If the equilibrium object is not an input, let create it
    if equ is None:
        equ = sf.EQU(shot, diag=diag, exp=exp, ed=ed)

    # equ.read_pfm()
    i = np.argmin(np.abs(equ.time - time))
    PFM = np.array(equ.pfm[:, :, i]).squeeze().astype(float)
    psipol = interpn((equ.Rmesh, equ.Zmesh), PFM, (Rin, zin), fill_value=0.0)

    return psipol


# -----------------------------------------------------------------------------
# --- Basic shot information
# -----------------------------------------------------------------------------
def get_shot_basics(shotnumber: int = None, diag: str = 'EQH',
                    exp: str = 'AUGD', edition: int = 0,
                    time: float = None):
    """
    Retrieves from the equilibrium reconstruction the basic data stored into
    a dictionary. Technically, it reads the SSQ from the equilibrium
    diagnostic.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number.
    :param  diag: Equilibrium diagnostic. By default EQH.
    :param  exp: Experiment where the data is stored.
    :param  edition: Edition of the shotfile.
    :param  time: time interval to retrieve. If it is a single value, only the
    appropriate data point will be retrieved. If None, all the data points are
    obtained.
    """
    # Checking the inputs.
    new_equ_opened = False
    try:
        sfo = sf.SFREAD(diag, shotnumber, experiment=exp, edition=edition)
        new_equ_opened = True
    except:
        raise errors.DatabaseError('EQU shotfile cannot be opened.')

    # Deactivate the nasty warnings for a while.
    eqh_time = np.asarray(sfo('time'))  # Time data.

    # Checking the time data.
    if time is not None:
        time = np.atleast_1d(time)

    nt = len(eqh_time)
    if time is None:
        t0 = 0
        t1 = nt
    elif len(time) == 1:
        t0 = np.abs(eqh_time.flatten() - time).argmin()
        t1 = t0+1
    else:
        t0 = np.abs(eqh_time.flatten() - time[0]).argmin() - 1
        t1 = np.abs(eqh_time.flatten() - time[-1]).argmin() + 1

    # Getting the names and the SSQ data.
    eqh_ssqnames = sfo('SSQnam')
    eqh_ssq = np.array(sfo('SSQ')).T

    # Unpacking the data.
    ssq = {}
    for jssq in range(eqh_ssq.shape[1]):
        tmp = b''.join(eqh_ssqnames[:, jssq]).strip()
        lbl = tmp.decode('utf-8')
        if lbl.strip() != '':
            tmp = b''.join(eqh_ssqnames[:, jssq]).strip()
            lbl = tmp.decode('utf-8')
            if lbl.strip() != '':
                decoded_key = lbl.replace('\x00', '')
                ssq[decoded_key] = eqh_ssq[t0:t1, jssq]

    # Reading from the equilibrium the magnetic flux at the axis and in the
    # separatrix.
    PFxx = sfo('PFxx').T
    ikCAT = np.argmin(abs(PFxx[1:, :] - PFxx[0, :]), axis=0) + 1
    ssq['psi_ax'] = PFxx[0, ...]
    ssq['psi_sp'] = [PFxx[iflux, ii] for ii, iflux in enumerate(ikCAT)]

    # Adding the time.
    ssq['time'] = np.atleast_1d(eqh_time[t0:t1])
    # --- Reading the plasma current.
    try:
        sfo = sf.SFREAD('MAG', shotnumber, experiment='AUGD', edition=edition)
    except:
        raise errors.DatabaseError('Error loading the MAG shotfile')

    # Getting the raw data.
    ipa_raw = sfo('Ipa')
    ipa = np.array(ipa_raw)
    ipa_time = np.array(sfo('T-MAG-1'))

    # Getting the calibration.
    parset = sfo.getparset('06ULID12')
    multi = parset['MULTIA00']
    shift = parset['SHIFTB00']

    ssq['ip'] = ipa * multi + shift  # This provides the current in A.
    ssq['ip'] *= 1.0e-6
    ssq['iptime'] = ipa_time

    # --- Getting the magnetic field at the axis.
    try:
        sfo = sf.SFREAD('MAI', shotnumber, experiment='AUGD',
                        edition=edition)
    except:
        raise errors.DatabaseError('MAI shotfile could not be loaded!')

    # Getting toroidal field.
    btf_sf = sfo('BTF')
    btf = np.array(btf_sf)
    btf_time = np.array(sfo('T-MAG-1'))

    # Getting the calibration.
    parset = sfo.getparset('14BTF')
    multi = parset['MULTIA00']
    shift = parset['SHIFTB00']

    ssq['bt0'] = multi*btf + shift
    ssq['bttime'] = btf_time

    return ssq


# -----------------------------------------------------------------------------
# --- q_profile
# -----------------------------------------------------------------------------
def get_q_profile(shot: int, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, sfo=None,
                  xArrayOutput: bool = True, **kwargs):
    """
    Reads from the database the q-profile as reconstrusted from an experiment.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  diag: Diag for AUG database, default EQH
    :param  exp: experiment, default AUGD
    :param  ed: edition, default 0 (last)
    :param  time: Array of times where we want to calculate the field
    :param  sf: shotfile accessing the data from the equilibrium.

    :return
    """
    if sfo is None:
        try:
            sfo = sf.SFREAD(diag, shot, experiment=exp, edition=ed)
        except:
            raise errors.DatabaseError(
                'Cannot open %05d:%s.%d to get the q-prof' % (shot, diag, ed))
    qpsi = sfo('Qpsi')
    pfl = sfo('PFL')
    timebasis = sfo('time')
    PFxx = sfo('PFxx')
    ikCAT = np.argmin(abs(PFxx[1:, :] - PFxx[0, :]), axis=0) + 1
    psi_ax = PFxx[0, ...]
    psi_edge = [PFxx[iflux, ii] for ii, iflux in enumerate(ikCAT)]
    psi_edge = np.tile(np.array(psi_edge), (pfl.shape[0], 1))
    rhop = np.sqrt((pfl - psi_ax)/(psi_edge-psi_ax)).squeeze()
    output = {}

    if time is not None:
        time = np.atleast_1d(time)

    if not xArrayOutput:
        if time is None:
            output = {
                'data': qpsi,
                'time': timebasis,
                'rhop': rhop
            }

        elif len(time) == 1:
            output = {
                'data': interp1d(timebasis, qpsi, axis=0)(time).squeeze(),
                'time': time.squeeze(),
                'rhop': interp1d(timebasis, rhop, axis=0)(time).squeeze()
            }
        elif len(time) == 2:
            t0, t1 = np.searchsorted(timebasis, time)
            output = {
                'data': qpsi[t0:t1, ...].squeeze(),
                'time': timebasis[t0:t1].squeeze(),
                'rhop': rhop[t0:t1, ...].squeeze(),
            }
        else:
            output = {
                'data': interp1d(timebasis, qpsi, axis=0)(time).squeeze(),
                'time': time.squeeze(),
                'rhop': interp1d(timebasis, rhop, axis=0)(time).squeeze(),
            }

        output['source'] = {
            'diagnostic': diag,
            'experiment': exp,
            'edition': ed,
            'pulseNumber': shot
        }
    else:
        output = xr.Dataset()
        found = False
        counter = 0
        while not found:
            try:
                jend = np.where(np.isnan(rhop[:, counter]))[0][0]
                found = True
            except IndexError:
                counter += 1
                if counter == rhop.shape[1]:
                    print(counter)
                    raise Exception('problem with the base')

        output['data'] = xr.DataArray(qpsi[:jend, :], dims=('rho', 't'),
                                      coords={'rho': rhop[:jend, 0],
                                      't': timebasis})
        output['data'].attrs['long_name'] = 'q'
        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'

        output.attrs['diag'] = diag
        output.attrs['exp'] = exp
        output.attrs['ed'] = ed
        output.attrs['shot'] = shot
    return output


def get_ECRH_traces(shot: int, time: float = None, ec_list: list = None):
    """
    Retrieves from the AUG database the ECRH timetraces with the power of the
    ECRH. The power and the injection angles are retrieved from the ECS
    shotfile while the actual position of the gyrotrons is obtained from TBM
    shotfile.

    Pablo Oyola - pablo.oyola@ipp.mpg.de


    :param  shot: Shot number
    :param  ed: edition, default 0 (last)
    :param  time: Array of times where we want to calculate the field. If None,
    the whole time array is retrieved.
    :param  ec_list: list with the ECRH gyrotrons to use. If None, all the
    gyrotrons are read.
    """

    if ec_list is None:
        ec_list = (1, 2, 3, 4, 5, 6, 7, 8)

    ec_list = np.atleast_1d(ec_list)

    try:
        sfecs = sf.SFREAD(shot, 'ECS', edition=0, experiment='AUGD')

        sftbm = sf.SFREAD(shot, 'TBM', edition=0, experiment='AUGD')
    except:
        raise errors.DatabaseError(
            'EC shotfiles cannot be opened for #%05d' % shot)

    output = dict()
    flag_first = False

    # --- Reading the data for all the gyrotrons in the list.
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for iecrh, ecrh_num in enumerate(ec_list):
        if ecrh_num <= 4:
            power_name = 'PG%d' % ecrh_num
        else:
            power_name = 'PG%dN' % (ecrh_num-4)

        # Getting the power of the gyrotron.
        power = sfecs(power_name)

        if np.all(power*1e-6 < ECRH_POWER_THRESHOLD):
            continue

        poloidal_angle_name = 'thpl-G%d' % ecrh_num
        toroidal_angle_name = 'phtr-G%d' % ecrh_num

        pol_ang = sfecs(poloidal_angle_name)
        tor_ang = sfecs(toroidal_angle_name)
        time_ang = sfecs('T-C')
        time_power = sfecs('T-B')

        if not flag_first:
            flag_first = True
            timebase = time_ang

        power_data = interp1d(time_power, power, bounds_error=False,
                              fill_value=0.0, assume_sorted=True)(timebase)

        polang_data = interp1d(time_ang, pol_ang, bounds_error=False,
                               fill_value=0.0, assume_sorted=True)(timebase)

        torang_data = interp1d(time_ang, tor_ang, bounds_error=False,
                               fill_value=0.0, assume_sorted=True)(timebase)

        del power

        output[int(ecrh_num)] = {
            'time': timebase,
            'power': power_data*1e-6,
            'pol_ang': polang_data,
            'tor_ang': torang_data,
        }

        # Getting the deposition position according to the RT controller.
        rhopol = sftbm('rhoout%d' % ecrh_num)
        Recrh = sftbm('R_out%d' % ecrh_num)
        zecrh = sftbm('z_out%d' % ecrh_num)
        rhoptime = sftbm('time_c')

        output[int(ecrh_num)]['time_pos'] = rhoptime
        output[int(ecrh_num)]['rhopol'] = rhopol
        output[int(ecrh_num)]['R'] = Recrh
        output[int(ecrh_num)]['z'] = zecrh

    # Reading the total power
    name = 'PECRH'
    pecrh = sfecs(name=name)
    output['total'] = {
        'time': timebase,
        'power': interp1d(time_power, pecrh, bounds_error=False,
                          fill_value=0.0)(timebase)*1.e-6
    }
    return output


def getECRH_total(shot: int, tBeg: float = None, tEnd: float = None,
                  xArrayOutput: bool = False):
    """
    Returns the total ECRH power from the ECS shotfile in AUG.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: shotnumber to get the ECRH power.
    :param  tBeg: initial time to get the timetrace. If None, the initial time
    stored in the shotfile will be returned.
    :param  tEnd: final time to get the timetrace. If None, the final time
    stored in the shotfile will be returned.
    """

    sf_ecs = sf.SFREAD('ECS', shot)
    if not sf_ecs.status:
        raise errors.DatabaseError(
            'Cannot get the ECS shotfile for #%05d' % shot)

    pecrh = sf_ecs(name='PECRH')
    time = sf_ecs.gettimebase('PECRH')


    if tBeg is None:
        t0 = 0
    else:
        t0 = np.abs(time - tBeg).argmin()

    if tEnd is None:
        t1 = len(time)
    else:
        t1 = np.abs(time - tEnd).argmin()

    # cutting the data to the desired time range.
    pecrh = pecrh[t0:t1]
    time = time[t0:t1]

    if xArrayOutput:
        output = xr.DataArray(pecrh/1.0e6, dims='t', coords={'t': time})
        output.attrs['long_name'] = '$P_{ECRH}$'
        output.attrs['units'] = 'MW'
        output.attrs['diag'] = 'ECS'
        output.attrs['signal'] = 'PECRH'
    else:
        output = {
            'power': pecrh,
            'time': time
        }

    return output


def getPrad_total(shot: int, tBeg: float = None, tEnd: float = None):
    """
    Return the total radiated power from the BPD shotfile in AUG.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: shotnumber to get the ECRH power.
    :param  tBeg: initial time to get the timetrace. If None, the initial time
    stored in the shotfile will be returned.
    :param  tEnd: final time to get the timetrace. If None, the final time
    stored in the shotfile will be returned.
    """

    sf_bpd = sf.SFREAD('BPD', shot)
    if not sf_bpd.status:
        raise errors.DatabaseError(
            'Cannot get the BPD shotfile for #%05d' % shot)

    prad = sf_bpd(name='Pradtot')
    time = sf_bpd.gettimebase('Pradtot')

    if tBeg is None:
        t0 = 0
    else:
        t0 = np.abs(time - tBeg).argmin()

    if tEnd is None:
        t1 = len(time)
    else:
        t1 = np.abs(time - tEnd).argmin()

    # cutting the data to the desired time range.
    prad = prad[t0:t1]
    time = time[t0:t1]

    output = {
        'power': prad,
        'time': time
    }

    return output
