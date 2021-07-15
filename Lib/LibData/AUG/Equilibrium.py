"""Routines for the magnetic equilibrium"""
import dd                # Module to load shotfiles
import numpy as np
import map_equ as meq    # Module to map the equilibrium
from scipy.interpolate import interpn
import warnings


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
    Wrap to get AUG normalised radius

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
    eqh_time = np.asarray(sf(name='time').data)  # Time data.

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
        tmp = b''.join(eqh_ssqnames[jssq, :]).strip()
        lbl = tmp.decode('utf8')
        if lbl.strip() != '':
            ssq[lbl] = eqh_ssq[t0:t1, jssq]

    if new_equ_opened:
        sf.close()

    # Adding the time.
    ssq['time'] = eqh_time[t0:t1]

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
