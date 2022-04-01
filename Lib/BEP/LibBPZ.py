"""
BPZ shotfile library.

This library contains routines to read and plot the data stored in the shotfile
BPZ. This shotfile contains the data as fitted by SPEC_FIT (by R. Dux).

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import matplotlib.pyplot as plt
import Lib
import dd
import warnings

# BPZ_SIGNAL_NAMES = ('L_E1', 'L_E2', 'L_E3', 'D_I', 'C_II', 'cont',
#                     'chisq', 'R', 'Z', 'fieldang', 'flxsfang', 'radef',
#                     'bnr1', 'bnr2', 'bnr3', 'sfieldan', 'sflxsfan', 'beamvolt',
#                     'gamw_dal', 'ph_at_E1', 'ph_at_E2', 'ph_at_E3')

BPZ_SIGNAL_NAMES = ('R', 'Z', 'fieldang', 'flxsfang', 'radef')

def readBPZ(shotnumber: int, time: float=None, exp: str='AUGD', edition=0,
            sf=None):
    """
    Reads the BPZ shotfile and stores the data into a dictionary that is sent
    back as an output. If 'sf' is provided, then all the shotfile info will
    be disregarded.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: shotnumber to open the shotfile.
    @param time: time window [t0, t1] to read the shotfile. If None, the whole
    time basis is read into a variable.
    @param exp: experiment where the shotfile is stored. By default, taken from
    AUGD.
    @param edition: shotfile edition. If None, the latest will be read.
    @param sf: shotfile handler. If this is provided, then the others are
    ignored.
    """

    # If the shotfile is not provided, then we open one temporarily.
    sf_new = False
    if sf is None:
        try:
            sf = dd.shotfile(diagnostic='BPZ', experiment=exp,
                             pulseNumber=shotnumber, edition=edition)
        except:
            raise Exception('Cannot read the BPZ shotfile for #%05d'%shotnumber)
        sf_new = True

    # Reading the timebase.
    try:
        timebase = sf(name='TIME')

    except:
        raise Exception('Cannot retrieve the time basis from shotfile')

    if time != None:
        t0, t1 = np.searchsorted(timebase.data, time)
    else:
        t0 = 0
        t1 = len(timebase.data)

    # Reading one by one the data from the shotfile.
    output = dict()
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    for ikey in BPZ_SIGNAL_NAMES:
        tmp = sf(name=ikey)

        output[ikey] = { 'data': tmp.data[t0:t1, ...],
                         'desc': tmp.header.text.decode('utf-8'),
                         'units': sf.getInfo(name=ikey).units.decode('utf-8')
                       }
    warnings.filterwarnings('default')
    if sf_new:
        sf.close()

    output['time'] = timebase[t0:t1].copy()

    # The next output will allow to use **diag in dd.shotfile for easy
    # access.
    output['diagnostic'] = { 'diagnostic': 'BPZ',
                             'edition': edition,
                             'experiment': exp,
                             'shotnumber': shotnumber
                           }

    return output


def plotAngle_BPZ(data: dict, which: str='tor', time: float=None, ax=None,
                  fig=None, ax_opts: dict={}, line_opts: dict={}):
    """
    Plots the time evolution of the angle as computed by the BEP fiting
    routines. Two angles are accepted:
    1. 'tor' -> arctan(Bpol/Btor) ~ q-profile
    2. 'pol' -> arctan(Bz/Br)     ~ Magnetic poloidal angle.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param data: dictionary with the data read from the BPZ shotfile.
    @param which: 'tor' or 'pol' referring to either poloidal angle or poloidal
    magnetic angle.
    @param time: time window to plot. If None, all the time window is used.
    @param ax: axis to plot the data. If None, new one will be created.
    @param fig: figure handler. If None, gcf() is used to retrieve it.
    @param ax_opts: dictionary with options to tune the axis.
    @param line_opts: dictionary with the options to plot the lines. Color will
    be fully disregarded!
    """

    if ('fieldang' not in data) and (which == 'tor'):
        raise ValueError('Toroidal angle not found in the data')

    if ('flxsfang' not in data) and (which == 'pol'):
        raise ValueError('Poloidal angle not found in the data')

    if 'time' not in data:
        raise ValueError('The timebase is not in the input data')

    # Getting the time limits according to the inputs
    if time != None:
        t0, t1 = np.searchsorted(data['time'], time)
    else:
        t0 = 0
        t1 = len(data['time'])

    ax_was_none = False
    if ax is None:
        fig, ax = plt.subplots(1)
        ax_was_none = True

    if fig is None:
        fig = plt.gcf()

    if 'color' in line_opts:
        line_opts.pop('color')

    if 'linewidth' not in line_opts:
        line_opts['linewidth'] = 2.0

    fieldname = { 'tor': 'fieldang',
                  'pol': 'flxsfang'
                }.get(which)

    if 'label' in line_opts:
        original_label = line_opts['label']
    else:
        original_label = ''
    # Plotting.
    for ii in range(data[fieldname]['data'].shape[1]):
        line_opts['label'] = original_label + 'LOS %d'%(ii+1)

        plt.plot(data['time'][t0:t1], data[fieldname]['data'][t0:t1, ii],
                 **line_opts)

    if ax_was_none:
        ax_opts['xlabel'] = 'Time [s]'
        ax_opts['ylabel'] = data[fieldname]['desc']

        ax=Lib.plt.axis_beauty(ax=ax, param_dict=ax_opts)

        ax.legend()

    return ax

def getAngle_from_EQ(shotnumber: int, Rin: float=None, Zin: float=None,
                     time: float=None, diag='EQH',
                     exp: str='AUGD', edition: int=0, exp_BPZ: str='AUGD'):

    """
    Computes the magnetic angles from the equilibrium diagnostics. It will
    compute the equivalent angles to the ones stored in the BPZ.

    1. 'tor' -> arctan(Bpol/Btor) ~ q-profile
    2. 'pol' -> arctan(Bz/Br)     ~ Magnetic poloidal angle.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    # Get the nearest BPZ shot to read in the position of the BEP LOS
    # intersections.
    if Rin is None or Zin is None:
        cal_shot = dd.getLastShotNumber(diagnostic=b'BPZ',
                                        pulseNumber=shotnumber,
                                        experiment=b'AUGD')

        sf = dd.shotfile(diagnostic='BPZ', pulseNumber=cal_shot,
                         experiment=exp_BPZ, edition=0)

        Rin = sf('R').data
        Zin = sf('R').data
        time = sf('TIME').data
        sf.close()

    Rin = np.atleast_1d(Rin)
    Zin = np.atleast_1d(Zin)
    time = np.atleast_1d(time)

    grr, gtt  = np.meshgrid(Rin, time)
    gzz, gtt  = np.meshgrid(Zin, time)

    br, bz, btor, _ = Lib.dat.get_mag_field(shot=shotnumber,
                                            Rin=grr,
                                            zin=gzz,
                                            diag=diag, exp=exp, ed=edition,
                                            time=time)

    bpol = np.sqrt(br**2.0 + bz**2.0)

    output = {  'fieldang': { 'data': np.arctan2(bpol, btor),
                              'desc': 'Field Line Angle ATAN(Bp,Bt)',
                              'units': 'rad'
                            },
                'flxsfang': { 'data': np.arctan2(bz, br),
                              'desc': 'FLux Surface Angle ATAN(Bz,Br) ',
                              'units': 'rad'
                            },
                'R': { 'data': Rin,
                       'desc': 'Major radius',
                       'units': 'm'
                     },
                'Z': { 'data': Zin,
                       'desc': 'Major radius',
                       'units': 'm'
                     },
                'time': time.copy(),
                'diag': { 'diagnostic': diag,  # This allows **diag in dd.shotfile.
                          'edition': edition,
                          'experiment': exp,
                          'shotnumber': shotnumber
                        }
             }

    return output

def getBpol_BPZ(shotnumber: int, time: float=None, exp_bpz: str='AUGD',
                ed_bpz: int=0):
    """
    Computes the poloidal magnetic field using the BPZ computed angles and the
    toroidal field as provided by the equilibrium data (which we assume is
    sufficiently reliable).

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: shot number of retrieve the experimental data.
    @param time: time window to compute the bpol.
    @param exp_bpz: experiment where the BPZ fit is saved.
    @param ed_bpz: edition of the BPZ shotfile to be opened.
    """

    bpz_data = readBPZ(shotnumber=shotnumber, time=time, exp=exp_bpz,
                       edition=ed_bpz)
    time = bpz_data['time']
    nR_bep = bpz_data['R']['data'].size
    # --- Retrieving the Btor from the equilibrium.
    R, Z = bpz_data['R']['data'], bpz_data['Z']['data']

    _, _, bt, _ =  Lib.dat.get_mag_field(shot=shotnumber,
                                         Rin=R.flatten(),
                                         zin=Z.flatten(),
                                         time=time.flatten())
    del _

    bt = np.reshape(bt, ( len(time), nR_bep))
    print(bt.shape, bpz_data['fieldang']['data'].shape)

    output = { 'R': bpz_data['R']['data'],
               'z': bpz_data['Z']['data'],
               'time': time,
               'bt': bt,
               'bp': bt*np.tan(bpz_data['fieldang']['data'])
             }

    return output


def getRhop_BPZ(shotnumber: int,  time: float=None,
                exp_bpz: str='AUGD', ed_bpz: int=0):
    """
    Computes rhopol value of each of the line of sights of BEP for a given
    shotfile using the standard equilibrium reconstruction.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: shot number of retrieve the experimental data.
    @param time: time window to compute the bpol.
    @param exp_bpz: experiment where the BPZ fit is saved.
    @param ed_bpz: edition of the BPZ shotfile to be opened.
    """
    bpz_data = readBPZ(shotnumber=shotnumber, time=time, exp=exp_bpz,
                       edition=ed_bpz)
    time = bpz_data['time']
    nR_bep = bpz_data['R']['data'].size
    # --- Retrieving the Btor from the equilibrium.
    R, Z = bpz_data['R']['data'], bpz_data['Z']['data']

    rhop =  Lib.dat.get_rho(shot=shotnumber, Rin=R.flatten(), zin=Z.flatten(),
                            time=time.flatten())

    rhop = np.reshape(rhop, ( len(time), nR_bep))

    output = { 'R': bpz_data['R']['data'],
               'z': bpz_data['Z']['data'],
               'time': time,
               'rhop': rhop
             }

    return output