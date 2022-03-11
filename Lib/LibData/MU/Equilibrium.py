"""Routines for the magnetic equilibrium"""
import warnings
import numpy as np
from scipy.interpolate import interpn, interp1d


ECRH_POWER_THRESHOLD = 0.05  # Threshold to consider ECRH on [MW]


# def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
#                   ed: int = 0, time: float = None, equ=None, **kwargs):
#     """
#     Wrapp to get AUG magnetic field
#
#     Jose Rueda: jrrueda@us.es
#
#     Note: No extra arguments are expected, **kwargs is just included for
#     compatibility of the call to this method in other databases (machines)
#
#     @param shot: Shot number
#     @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
#     @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
#     @param diag: Diag for AUG database, default EQH
#     @param exp: experiment, default AUGD
#     @param ed: edition, default 0 (last)
#     @param time: Array of times where we want to calculate the field (the
#     field would be calculated in a time as close as possible to this
#     @param equ: equilibrium object from the library aug_sfutils
#
#     @return br: Radial magnetic field (nt, nrz_in), [T]
#     @return bz: z magnetic field (nt, nrz_in), [T]
#     @return bt: toroidal magnetic field (nt, nrz_in), [T]
#     @return bp: poloidal magnetic field (nt, nrz_in), [T]
#     """
#     # If the equilibrium object is not an input, let create it
#     # created = False
#     if equ is None:
#         equ = sf.EQU(shot, diag='EQI', ed=1)
#         # equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
#         # created = True
#     # Now calculate the field
#     # br, bz, bt = equ.rz2brzt(Rin, zin, t_in=time)
#     br, bz, bt = sf.rz2brzt(equ, r_in=Rin, z_in=zin, t_in=time)
#     bp = np.hypot(br, bz)
#     # # If we opened the equilibrium object, let's close it
#     # if created:  # no need with the new library
#     #     equ.Close()
#     return br, bz, bt, bp
def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, equ = None):
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
    # If the equilibrium object is not an input, let's create it
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
    Wrapp to get AUG normalised radius.

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


def get_rho2rz(shot: int, flxlabel: float, diag: str = 'EQH', exp: str = 'AUGD',
               ed: int = 0, time: float = None, coord_out: str = 'rho_pol',
               equ=None):
    """
    Gets the curves (R, z) associated to a given flux surface.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param flxlabel: flux surface label.
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @param coord_out: the desired rho coordinate, default rho_pol
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True

    R, z = equ.rho2rz(t_in=time, rho_in=flxlabel, coord_in=coord_out,
                      all_lines=False)

    if time is None:
        tout = equ.t_eq
    else:
        tout = time

    if created:
        equ.Close()

    return R, z, tout



def get_psipol(shot: int, Rin, zin, diag='EQH', exp: str = 'AUGD',
               ed: int = 0, time: float = None, equ=None):
    """
    Wrap to get AUG poloidal flux field

    Jose Rueda: jrrueda@us.es
    ft.
    Pablo Oyola - pablo.oyola@ipp.mpg.de

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
    psipol = interpn((equ.Rmesh, equ.Zmesh), PFM, (Rin, zin), fill_value=0.0)

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
    eqh_time = np.asarray(sf(name='time').data)  # Time data.

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
    eqh_ssqnames = sf.GetSignal(name='SSQnam')
    eqh_ssq = sf.GetSignal(name='SSQ')
    warnings.filterwarnings('default')

    # Unpacking the data.
    ssq = dict()
    for jssq in range(eqh_ssq.shape[1]):
        tmp = b''.join(eqh_ssqnames[jssq, :]).strip()
        lbl = tmp.decode('utf8')
        if lbl.strip() != '':
            ssq[lbl] = eqh_ssq[t0:t1, jssq]

    # Reading from the equilibrium the magnetic flux at the axis and in the
    # separatrix.
    PFxx = sf.GetSignal('PFxx').T
    ikCAT = np.argmin(abs(PFxx[1:, :] - PFxx[0, :]), axis=0) + 1
    ssq['psi_ax'] = PFxx[0, ...]
    ssq['psi_sp'] =  [PFxx[iflux, ii] for ii, iflux in enumerate(ikCAT)]

    if new_equ_opened:
        sf.close()

    # Adding the time.
    ssq['time'] = np.atleast_1d(eqh_time[t0:t1])
    # --- Reading the plasma current.
    try:
        sf = dd.shotfile(pulseNumber=shotnumber, diagnostic='MAG',
                         experiment='AUGD', edition=0)
    except:
        raise Exception('Error loading the MAG shotfile')

    # Getting the raw data.
    ipa_raw = sf(name='Ipa', tBegin=ssq['time'][0], tEnd=ssq['time'][-1])
    ipa = ipa_raw.data
    ipa_time = ipa_raw.time

    # Getting the calibration.
    multi = sf.getParameter('06ULID12', 'MULTIA00').data.astype(dtype=float)
    shift = sf.getParameter('06ULID12', 'SHIFTB00').data.astype(dtype=float)

    ssq['ip'] = ipa * multi + shift  # This provides the current in A.
    ssq['ip'] *= 1.0e-6
    ssq['iptime'] = ipa_time

    # Close the shotfile.
    sf.close()

    # --- Getting the magnetic field at the axis.
    try:
        sf = dd.shotfile(pulseNumber=shotnumber, experiment='AUGD',
                         diagnostic='MAI', edition=0)
    except:
        raise Exception('MAI shotfile could not be loaded!')

    # Getting toroidal field.
    btf_sf = sf(name='BTF', tBegin=ssq['time'][0], tEnd=ssq['time'][-1])
    btf = btf_sf.data
    btf_time = btf_sf.time

    # Getting the calibration.
    multi = sf.getParameter('14BTF', 'MULTIA00').data.astype(dtype=float)
    shift = sf.getParameter('14BTF', 'SHIFTB00').data.astype(dtype=float)

    ssq['bt0'] = multi*btf + shift
    ssq['bttime'] = btf_time

    # Close the shotfile
    sf.close()
    return ssq


def get_q_profile(shot: int, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, sf=None):
    """
    Reads from the database the q-profile as reconstrusted from an experiment.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field
    @param sf: shotfile accessing the data from the equilibrium.

    @return
    """

    sf_new = False
    if sf is None:
        sf_new = True
        try:
            sf = dd.shotfile(diagnostic=diag, experiment=exp,
                             pulseNumber=shot, edition=ed)
        except:
            raise Exception('Cannot open %05d:%s.%d to get the q-prof'%(shot,
                            diag, ed))


    qpsi = sf(name='Qpsi').data
    pfl  = sf(name='PFL').data
    timebasis = sf(name='time').data

    PFxx = sf.GetSignal('PFxx').T
    ikCAT = np.argmin(abs(PFxx[1:, :] - PFxx[0, :]), axis=0) + 1
    psi_ax = np.tile(PFxx[0, ...], (pfl.shape[1], 1)).T
    psi_edge =  [PFxx[iflux, ii] for ii, iflux in enumerate(ikCAT)]
    psi_edge = np.tile(np.array(psi_edge), (pfl.shape[1], 1)).T

    rhop = np.sqrt((pfl - psi_ax)/(psi_edge-psi_ax)).squeeze()
    output = {}

    if time is not None:
        time = np.atleast_1d(time)

    if time is None:
        output = { 'data': qpsi,
                   'time': timebasis,
                   'rhop': rhop
                 }

    elif len(time) == 1:
        output = { 'data': interp1d(timebasis, qpsi, axis=0)(time).squeeze(),
                   'time': time.squeeze(),
                   'rhop': interp1d(timebasis, rhop, axis=0)(time).squeeze()
                 }
    elif len(time) == 2:
        t0, t1 = np.searchsorted(timebasis, time)
        output = { 'data': qpsi[t0:t1, ...].squeeze(),
                   'time': timebasis[t0:t1].squeeze(),
                   'rhop': rhop[t0:t1, ...].squeeze(),
                 }
    else:
         output = { 'data': interp1d(timebasis, qpsi, axis=0)(time).squeeze(),
                    'time': time.squeeze(),
                    'rhop': interp1d(timebasis, rhop, axis=0)(time).squeeze(),
                 }

    if sf_new:
        sf.close()

    output['source'] = { 'diagnostic': diag,
                         'experiment': exp,
                         'edition': ed,
                         'pulseNumber': shot
                       }

    return output

def get_ECRH_traces(shot: int, time: float=None, ec_list: list=None):
    """
    Retrieves from the AUG database the ECRH timetraces with the power of the
    ECRH. The power and the injection angles are retrieved from the ECS
    shotfile while the actual position of the gyrotrons is obtained from TBM
    shotfile.

    Pablo Oyola - pablo.oyola@ipp.mpg.de


    @param shot: Shot number
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field. If None,
    the whole time array is retrieved.
    @param ec_list: list with the ECRH gyrotrons to use. If None, all the
    gyrotrons are read.
    """

    if ec_list is None:
        ec_list = (1, 2, 3, 4, 5, 6, 7, 8)

    ec_list = np.atleast_1d(ec_list)

    try:
        sfecs = dd.shotfile(diagnostic='ECS', pulseNumber=shot,
                            edition=0, experiment='AUGD')

        sftbm = dd.shotfile(diagnostic='TBM', pulseNumber=shot,
                            edition=0, experiment='AUGD')
    except:
        raise Exception('EC shotfiles cannot be opened for #%05d' % shot)

    output = dict()
    flag_first = False

    # --- Reading the data for all the gyrotrons in the list.
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for iecrh, ecrh_num in enumerate(ec_list):
        if ecrh_num <= 4:
            power_name = 'PG%d'%ecrh_num
        else:
            power_name = 'PG%dN'%(ecrh_num-4)

        # Getting the power of the gyrotron.
        power = sfecs(power_name)

        if np.all(power.data*1e-6 < ECRH_POWER_THRESHOLD):
            continue


        poloidal_angle_name = 'thpl-G%d' % ecrh_num
        toroidal_angle_name = 'phtr-G%d' % ecrh_num

        pol_ang = sfecs(poloidal_angle_name)
        tor_ang = sfecs(toroidal_angle_name)

        if not flag_first:
            flag_first = True
            timebase = pol_ang.time

        power_data = interp1d(power.time, power.data, bounds_error=False,
                              fill_value=0.0, assume_sorted=True)(timebase)

        polang_data = interp1d(pol_ang.time, pol_ang.data, bounds_error=False,
                               fill_value=0.0, assume_sorted=True)(timebase)

        torang_data = interp1d(tor_ang.time, tor_ang.data, bounds_error=False,
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

        output[int(ecrh_num)]['time_pos'] = rhopol.time
        output[int(ecrh_num)]['rhopol'] = rhopol.data
        output[int(ecrh_num)]['R'] = Recrh.time
        output[int(ecrh_num)]['z'] = zecrh.time

    # Reading the total power
    name = 'PECRH'
    pecrh = sfecs(name=name)
    output['total'] = { 'time': timebase,
                        'power': interp1d(pecrh.time, pecrh.data,
                                          bounds_error=False,
                                          fill_value=0.0)(timebase)*1.e-6
                      }

    warnings.filterwarnings('default', category=RuntimeWarning)

    return output
