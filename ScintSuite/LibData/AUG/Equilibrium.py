"""
Routines for the magnetic equilibrium
"""
import warnings
import numpy as np
import xarray as xr
import aug_sfutils as sf
import ScintSuite.errors as errors
from scipy.interpolate import interpn, interp1d

from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

xr.set_options(keep_attrs=True)
from .Misc import to_dict_with_metadata


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
            coord_out: str = 'rho_pol', **kwargs):
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
# --- Basic shot information (update 03/10/2026)
# -----------------------------------------------------------------------------
def get_shot_basics(shot: int = None, diag: str = 'EQH',
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
        sfo = sf.SFREAD(diag, shot, experiment=exp, edition=edition)
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
        sfo = sf.SFREAD('MAG', shot, experiment='AUGD', edition=edition)
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
        sfo = sf.SFREAD('MAI', shot, experiment='AUGD',
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

def get_Ip(shot, plot = False, xArrayOutput: bool = True):
    MAG = sf.SFREAD(shot, 'MAG')
    name = 'Ipa'
    I = np.array(MAG(name), dtype='f4') / 1.0e6
    t = np.array(MAG.gettimebase(name), dtype='f4')
    obj = xr.DataArray(I, dims=['t'], 
                       coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})},
                       attrs={'units': 'MA', 'long_name': '$I_p$', 
                              'shot': shot, 'diag': 'MAG', 'signal': name}
    )
    if plot:
        fig, ax = plt.subplots()
        obj.plot(ax=ax)
    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_Bt(shot, plot = False, xArrayOutput: bool = True):
    MAI = sf.SFREAD(shot, 'MAI')
    name = 'BTF'
    B = np.array(MAI(name), dtype='f4') * -1.0
    t = np.array(MAI.gettimebase(name), dtype='f4')
    obj = xr.DataArray(B, dims=['t'],
                       coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})}, 
                       attrs={'units': 'T', 'long_name': '$B_t$', 
                              'shot': shot, 'diag': 'MAI', 'signal': name        }
    )
    if plot:
        fig, ax = plt.subplots()
        obj.plot(ax=ax)
    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

# -----------------------------------------------------------------------------
# --- q_profile (update 03/10/2026)
# -----------------------------------------------------------------------------
def get_q_profile(shot, diag: str = 'EQH', plot = False, 
                  xArrayOutput: bool = True):
    """
    Reads from the database the q-profile as reconstrusted from an experiment.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    Alex Reyner: areyner@us.es

    :param  shot: Shot number
    :param  diag: Diag for AUG database, default EQH

    :return
    """

    diags_to_try = ['FPG', 'EQH'] if diag == 'FPG' else ['EQH', 'FPG']
    errors = {}
    for d in diags_to_try:
        try:
            if d == 'EQH': obj = get_qprof_EQH(shot)
            elif d == 'FPG': obj = get_qprof_FPG(shot)
            break
        except Exception as e:
            errors[d] = e

    if plot:
        fig, ax = plt.subplots()
        if d == 'FPG':
            obj.to_array().plot.line(x='t', hue='variable')
        elif d == 'EQH':
            obj.plot.imshow(robust=True)
    
    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_qprof_EQH(shot):

    EQH = sf.SFREAD('EQH', shot)
    qpsi = np.array(EQH('Qpsi'), dtype='f4') *-1
    pfl = np.array(EQH('PFL'), dtype='f4')
    t = np.array(EQH('time'), dtype='f4')
    PFxx = np.array(EQH('PFxx'), dtype='f4')

    ikCAT = np.argmin(np.abs(PFxx[1:, :] - PFxx[0, :]), axis=0) + 1
    psi_ax = PFxx[0, :]
    psi_edge = PFxx[ikCAT, np.arange(PFxx.shape[1])]

    rhop = np.sqrt((pfl - psi_ax[None, :]) / 
                   (psi_edge[None, :] - psi_ax[None, :])).squeeze()

    nan_mask = np.isnan(rhop)
    if nan_mask.any():
        valid_counts = (~nan_mask).all(axis=1)
        jend = (np.argmin(valid_counts) 
                if not valid_counts.all() else rhop.shape[0])
    else:
        jend = rhop.shape[0]

    if jend == 0:
        raise RuntimeError(f"Invalid magnetic grid reconstruction in EQH for shot {shot}")

    rho_grid = rhop[:jend, 0]

    obj = xr.DataArray(qpsi[:jend, :], dims=['rho', 't'],
        coords={'rho': ('rho', rho_grid, {'long_name': r'$\rho_p$', 'units': ''}), 
                't': ('t', t, {'long_name': 'Time', 'units': 's'}),},
        attrs={'long_name': 'Safety factor', 'shot': shot, 'diag': 'EQH',},
    )

    return obj

def get_qprof_FPG(shot):
    FPG = sf.SFREAD('FPG', shot)
    t = np.array(FPG.gettimebase('q95'), dtype='f4')
    rhos = np.array([0, 25, 50, 75, 95])
    obj = xr.Dataset(coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})},
                     attrs={'shot': shot, 'diag': 'FPG'}
    )
    for rho in rhos:
        name = f'q{rho}'
        data = np.array(FPG(name), dtype='f4') * -1.0

        obj[name] = (['t'], data, 
                     {'long_name': rf'$q_{{{rho}}}$', 'units': '', 
                      'signal': name},
                )
    return obj

