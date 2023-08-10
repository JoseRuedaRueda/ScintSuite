"""Magnetic coils data"""

import os
import logging
import numpy as np
import aug_sfutils as SF
import ScintSuite.errors as errors
import ScintSuite.LibData.AUG.DiagParam as params

# import dd                # Module to load shotfiles
from tqdm import tqdm
from typing import Optional
from ScintSuite._Paths import Path
from scipy.interpolate import interp1d

try:
    import requests
except ModuleNotFoundError:
    print('request package is not installed. Install it to read '
          + 'the magnetic phase corrections.')

try:
    import re
except ModuleNotFoundError:
    print('re package is not installed. Install it to read '
          + 'the magnetic phase corrections.')
try:
    import shutil
except ModuleNotFoundError:
    print('shutil package is not installed. Install it to read '
          + 'the magnetic phase corrections.')

# ----------------------------------------------------------------------------
# %% Auxiliary objects
# ----------------------------------------------------------------------------
paths = Path(machine='AUG')
logger = logging.getLogger('ScintSuite.Magnetics')

# ----------------------------------------------------------------------------
# --- Coils corrections routines.
# ----------------------------------------------------------------------------
def magneticPhaseCorrection(coilnumber: int, coilgrp: str, freq: float = None,
                            shotnumber: int = None):
    """
    Read impedance correction for phase calculation of the magnetic
    pick-up coils.

    Taken from mode_determinaton.py by
        Felician Mink, Marcus Roth & Markus Wappl

    :param  coilnumber: number of the coils within its group.
    :param  coilgrp: group name of the coil, like 'B31', 'B17', ...
    :param  freq: frequency array to evaluate the correction (in kHz).
    :param  shotnumber: shotnumber for the coils. The correction is taken in
    different times.
    """
    if freq is None:
        freq = np.linspace(start=0.0, stop=500.0, num=256, dtype=float)

    if shotnumber is None:
        shotnumber = 33724 + 1

    imped_phase = np.zeros((len(freq), 2))

    magPath = paths.bcoils_phase_corr

    if not os.path.isdir(magPath):
        try:
            os.mkdir(magPath)
        except FileExistsError:
            raise Exception('Cannot create folder! Admin privileges needed?!')
        print('Downloading AUG phase corrections')

        url = 'https://datashare.mpcdf.mpg.de/s/FiqRIixNMb82HTq/download'

        r = requests.get(url, allow_redirects=True)
        filename = re.findall('filename=(.+)',
                              r.headers.get('content-disposition'))[0]

        filename = filename.replace('\"', '')

        with open(filename, 'wb') as fid:
            fid.write(r.content)

        print('Done! Unpacking...')
        shutil.unpack_archive(filename, magPath)
        os.remove(filename)

    if coilgrp not in ('B31', 'C07', 'B17'):
        print('Coils not supported for phase-correction')
        output = {
            'freq': freq,
            'phase': np.ones(np.atleast_1d(freq).shape),
            'interp': lambda x: 1.0
        }
        return output

    coilfullname = '%s-%02d' % (coilgrp, coilnumber)

    if shotnumber > 33724:
        path = os.path.join(magPath, 'felix_trans') + '/' + \
                coilfullname + ".txt"
    elif shotnumber > 31776:
        path = os.path.join(magPath, 'mink') + '/' + \
                coilfullname + ".txt"
    else:
        path = os.path.join(magPath, 'horvath') + '/' + \
                coilfullname + ".txt"
    try:
        with open(path, "r") as ins:
            array = []
            for line in ins:
                array.append(line)
        entries = len(array[0].split())
        datas = np.zeros((len(array), entries))
        for i in range(len(array)):
            split = array[i].split()
            for k in range(entries):
                datas[i, k] = split[k]
        if entries > 2:
            imped_phase[:, 0] = np.interp(freq, datas[:, 0]/1000., datas[:, 4])
            imped_phase[:, 1] = np.interp(freq, datas[:, 0]/1000., datas[:, 3])

            imped_phase[np.where(freq > 500)[0], 1] = datas[-1, 3]
            imped_phase[np.where(freq < 0)[0], 1] = datas[0, 3]

            imped_phase[np.where(freq > 500)[0], 0] = 1.0
            imped_phase[np.where(freq < 0)[0], 0] = 1.0
        else:
            imped_phase[:, 0] = np.ones(len(freq))
            imped_phase[:, 1] = np.interp(freq, datas[:, 0], datas[:, 1])

            imped_phase[np.where(freq > 500)[0], 1] = datas[-1, 1]
            imped_phase[np.where(freq < 0)[0], 1] = datas[0, 1]

    except ValueError:
        print('Transferfunction valueerror in %s' % coilfullname)
        imped_phase[:, 0] = 1.0
        imped_phase[:, 1] = 0.0
    except IOError:
        print('Transferfunction IOerror in %s' % coilfullname)
        imped_phase[:, 0] = 1.0
        imped_phase[:, 1] = 0.0

    output = {
        'freq': freq,
        'phase': imped_phase[:, 1],
        'interp': interp1d(freq, imped_phase[:, 1], kind='linear',
                            bounds_error=False, fill_value=0.0,
                            assume_sorted=True)
    }
    return output


def get_magnetics(shotnumber: int, coilNumber: int, coilGroup: str = 'B31',
                  timeWindow: Optional[list] = None):
    """
    Retrieve from the shot file the magnetic data information.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: Shot number to get the data.
    :param  coilNumber: Coil number in the coil array.
    :param  coilGroup: can be B31, B17, C09,... by default set to B31
    (ballooning coils)
    :param  timeWindow: Time window to get the magnetic data. If None, all the
    time window will be obtained.
    :return output: magnetic data (time traces and position.)
    """
    if shotnumber <= 33739:
        diag = 'MHA'
    else:
        diag = 'MHI'

    exp = 'AUGD'  # Only the AUGD data retrieve is supported with the dd.

    try:
        sf = SF.SFREAD(diag, shotnumber, experiment=exp,  edition=0)
    except:
        raise errors.DatabaseError(
            'Shotfile not existent for ' + diag + ' #' + str(shotnumber))

    try:
        # Getting the time base.
        time = sf('Time')
    except:
        raise errors.DatabaseError('Time base not available in shotfile!')

    if timeWindow is None:
        timeWindow = [time[0], time[-1]]

    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[1])

    name = '%s-%02d' % (coilGroup, coilNumber)
    try:
        # Getting the time base.
        mhi = sf(name)
    except:
        raise errors.DatabaseError(name+' not available in shotfile.')

    # --- Getting the calibration factors from the CMH shotfile.
    # In the new sf utils, there is no last shot option, so we put a loop
    flag = True
    shot2 = shotnumber
    while flag:
        sfcal = SF.SFREAD('CMH', shot2)
        if sfcal.status:
            flag = False
        else:
            shot2 += -1
    cal_name = 'C'+name
    cal = sfcal.getparset(cal_name)
    timebase = sf.gettimebase(name)
    t0 = np.abs(timebase-timeWindow[0]).argmin()
    t1 = np.abs(timebase-timeWindow[-1]).argmin()
    output = {
        'time': timebase[t0:t1],
        'data': np.array(mhi.data[t0:t1]),
        'R': cal['R'],
        'z': cal['z'],
        'phi': cal['phi'],
        'theta': cal['theta'],
        'area': cal['EffArea']
    }

    # --- Pick-up coils phase correction
    output['phase_corr'] = magneticPhaseCorrection(coilNumber, coilGroup,
                                                   shotnumber=shotnumber)

    return output


def get_magnetic_poloidal_grp(shotnumber: int, timeWindow: float,
                              coilGrp: int = None):
    """
    It retrieves from the database the data of a set of magnetic coils lying
    within the same phi-angle. The way to choose them is either providing the
    coils group as provided above or giving the phi angle, and the nearest set
    of coils will be obtained. For example:
        phi_range = 45º will provide the C07 coil group.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: Shotnumber to retrieve the magnetic coils.
    :param  timeWindow: time window to get the data.
    :param  coilGrp: 'C07', 'C09', 'B31-5_11', 'B31-32-38' depending upon the
    group that needs to be read from database.
    :param  phi_mag: each coil group is characterized by an approximate phi
    angle. If this is provided, the nearest group in phi is retrieved from the
    database. See variable @see{MHI_GROUP_APPR_PHI} to see which are the
    angles used.

    :return output: group of the magnetic coils.
    """
    # --- Parsing the inputs
    if coilGrp is None:
        raise errors.NotValidInput(
            'Either the coil-group or the angle must be provided.')

    # If the time window is a single float, we will make an array containing
    # at least a 5ms window.
    dt = 5e-3
    if len(timeWindow) == 1:
        timeWindow = np.array((timeWindow, timeWindow + dt))

    # --- Parsing the magnetic group input
    # Check if it is in the list of coils.
    if coilGrp not in params.mag_coils_grp2coilName:
        raise Exception('Coil group, %s, not found!' % coilGrp)

    # Getting the list with the coils names.
    coil_list = params.mag_coils_grp2coilName[coilGrp]
    numcoils = len(coil_list[1])

    # --- Opening the shotfile.
    if shotnumber <= 33739:
        diag = 'MHA'
    else:
        diag = 'MHI'

    exp = 'AUGD'  # Only the AUGD data retrieve is supported with the dd.

    try:
        sf = SF.SFREAD(diag, shotnumber, experiment=exp,  edition=0)
    except:
        raise errors.DatabaseError(
            'Shotfile not existent for ' + diag + ' #' + str(shotnumber))

    try:
        # Getting the time base.
        time = sf('Time')
    except:
        raise errors.DatabaseError('Time base not available in shotfile!')

    if timeWindow is None:
        timeWindow = [time[0], time[-1]]

    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[1])

    # --- Getting the calibration factors from the CMH shotfile.
    # Get the last shotnumber where the calibration is written.
    # In the new sf utils, there is no last shot option, so we put a loop
    flag = True
    shot2 = shotnumber
    while flag:
        sfcal = SF.SFREAD('CMH', shot2)
        if sfcal.status:
            flag = False
        else:
            shot2 += -1

    # --- Getting the coils data.
    output = {
        'phi': np.zeros((numcoils,)),
        'theta': np.zeros((numcoils,)),
        'dtheta': np.zeros((numcoils,)),
        'R': np.zeros((numcoils,)),
        'z': np.zeros((numcoils,)),
        'area': np.zeros((numcoils,)),
        'time': [],
        'data': [],
        'coilNumber': np.zeros((numcoils,)),
        'phase_correction': list()
    }

    jj = 0
    flags = np.zeros((numcoils,), dtype=bool)
    for ii in tqdm(np.arange(numcoils)):
        name = '%s-%02d' % (coil_list[0], coil_list[1][ii])
        cal_name = 'C'+name

        try:
            # Try to get the magnetic data.
            mhi = sf(name=name, tBegin=timeWindow[0], tEnd=timeWindow[-1])

            # Try to get the calibration.
            cal = sfcal.getparset(cal_name)
        except:
            print('Could not retrieve coils %s-%02d' %
                  (coil_list[0], coil_list[1][ii]))
            continue
        timebase = sf.gettimebase(name)
        t0 = np.abs(timebase-timeWindow[0]).argmin()
        t1 = np.abs(timebase-timeWindow[-1]).argmin()

        if jj == 0:
            output['time'] = timebase[t0:t1]
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

        # Appending the phase correction in case that it is needed.
        # --- Pick-up coils phase correction
        output['phase_corr'].append(magneticPhaseCorrection(coil_list[0],
                                                            coil_list[1][ii],
                                                            shotnumber))
        jj += 1

        del mhi
        del cal

    # --- All the coils have been read.
    # Removing the holes left by the coils that were not available in MHI.
    output['phi'] = output['phi'][flags]
    output['theta'] = output['theta'][flags]
    output['dtheta'] = output['dtheta'][flags]
    output['R'] = output['R'][flags]
    output['z'] = output['z'][flags]
    output['area'] = output['area'][flags]
    output['coilNumber'] = output['coilNumber'][flags]

    return output
