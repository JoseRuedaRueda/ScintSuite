"""Read from the data base."""
import Lib.LibData as aug
import Lib.LibPlotting as ssplt
import numpy as np
import matplotlib.pyplot as plt
import Lib.LibPaths as lpath
import os
import dd
import shutil
try:
    import netCDF4 as nc4
except ModuleNotFoundError:
    print('Yout cannot save or load BEB database')        


FCTSPH = 0.46926  # Counts per photon.
# -----------------------------------------------------------------------------
# --- Variables to read with the BEP.
# -----------------------------------------------------------------------------
beppget_path = '/afs/ipp/u/cxrs/BPZ/amd64_linux26/bepget'
bep_spectra_path = lpath.Path().ScintSuite+'Data/BEP/spectra'

spectra_nc4_names = ['time', 'lambda', 'spectra', 'losnam', 'neon']

# -----------------------------------------------------------------------------
# --- Function reading a plotting the calibrated signals.
# -----------------------------------------------------------------------------
def BEP_readnetcdf(filename: str, varName: str = None):
    """
    Reads the netCDF BEP data obtained beppget into a dictionary.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param filename: path where the file is contained to be read.
    @param varName: used only to get single data set from the file.
    """

    # --- Opening the dataset.
    rootgrp = nc4.Dataset(filename, 'r')

    # --- Checking whether a single variable is needed or all of them.
    read_all = False
    if varName is None:
        read_all = True

    if not read_all:
        if varName not in spectra_nc4_names:
            raise Exception('Variable is not in spectra netCDF4')

        try:
            varHandler = rootgrp.variables[varName]
        except:
            raise Exception('Variable not found in the netCDF4 - Corrupted?')


        output = { varName: varHandler[:] }

    else:
        output = dict()
        for ii in spectra_nc4_names:
            try:
                if rootgrp.variables[ii][:].dtype == '|S1':
                    tmp_list = list()
                    datatmp = rootgrp.variables[ii][:].data
                    for jj in range(datatmp.shape[0]):
                        tmp = (b''.join(datatmp[jj, :])).strip()
                        tmp_list.append(tmp.decode('utf8'))
                    output[ii] = tmp_list
                else:
                    output[ii] = np.array(rootgrp.variables[ii][:].data)
            except:
                raise Exception('Variable not found in the netCDF4 - Corrupted?')

    rootgrp.close()
    return output


def readBEP(shotnumber: int, time: float, experiment: str = 'AUGD',
            edition: int = 0, smear: bool = True):

    """
    Reads the BEP data from the shotnumber in the given interval (given that
    NBI6 was ON and NBI5 is OFF in that time window) for the input shotnumber.
    We will use here Ralph's bepget to gain access to the spectra already
    calibrated in wavelength.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: Shot number to retrieve the data.
    @param time: float array containing the time window to retrieve the data.
    @param experiment: experiment where the data is stored. Default to AUGD.
    @param edition: edition of the data.
    @param channels: float array containing the channels to read. If None, all
    will be read.
    """

    if smear:
        smear_s = 1
    else:
        smear_s = 0

    #--- Check if the shotfile is already stored into a netcdf-file.
    filename_tmp = '/tmp/all_spectra_BEP_%s'%os.getenv('USER')
    filename_db = '%s/%05d_%s'%(bep_spectra_path, shotnumber,
                                os.getenv('USER'))

    if (not os.path.isfile(filename_tmp)) and \
        (not os.path.isfile(filename_tmp)):
            command = '%s %s %05d %d'%(beppget_path, experiment,
                                        shotnumber, smear_s)
            os.system(command)
            shutil.copy2(src=filename_tmp, dst=filename_db)
    elif not os.path.isfile(filename_db):
        rootgrp = nc4.Dataset(filename_tmp, 'r')
        if rootgrp.shot != shotnumber:
            command = '%s %s %05d %d'%(beppget_path, experiment,
                                        shotnumber, smear_s)
            os.system(command)
        rootgrp.close()
        shutil.copy2(src=filename_tmp, dst=filename_db)

    # --- Reading the file into a dictionary.
    output = BEP_readnetcdf(filename=filename_db)

    # --- Masking the time.
    if len(time) == 1:
        t0 = np.abs(output['time'] - time).argmin()
        output['time'] = output['time'][t0:]
        output['spectra'] = output['spectra'][:, t0:, :]
    else:
        t0 = np.abs(output['time'] - time[0]).argmin()
        t1 = np.abs(output['time'] - time[1]).argmin()

        output['time'] = output['time'][t0:t1]
        output['spectra'] = output['spectra'][:, t0:t1, :]

    return output


def plotBEP(bepdata: dict, losname: str = 'BEP-1-3', avg: bool = True,
            ax=None, ax_options: dict={}, line_options:dict={},
            time: float = None):
    """
    Plots the BEP spectra for the input LOS.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param bepdata: dictionary with the BEP spectra.
    @param losname: name of the LOS to plot.
    @para avg: plot the average BEP signal in the given time window. If True,
    average is plot. Otherwise, each different time slice is plotted.
    @param ax: axis to plot. If None, ones will be created.
    @param fig: figure handler where the plot is. Otherwise, gcf is used.
    @param ax_options: dictionary with options for the axis.
    @param line_options: dictionary with options to plot.
    @param time: time window to plot. If None, all the input time window is
    used instead.
    """

    #--- Checking the inputs.
    if (losname not in bepdata['losnam']) and losname != 'neon':
        raise Exception('The LOS Name is not in the list')

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(1)

    if 'grid' not in ax_options:
        ax_options['grid'] = 'both'

    if 'fontsize' not in ax_options:
        ax_options['fontsize'] = 16

    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2


    if time is None:
        time = [bepdata['time'][0], bepdata['time'][-1]]

    t0, t1 = bepdata['time'].searchsorted(time)
    t1 += 1


    # --- Plotting the data.
    if losname == 'neon':
        if 'label' not in line_options:
            line_options['label'] = 'Neon lamp'
        else:
            line_options['label'] += ' - Neon lamp'

        plt.plot(bepdata['lambda'][0, :], bepdata['neon'],
                 **line_options)
    else:
        idx_channel = bepdata['losnam'].index(losname)
        lambdaval = bepdata['lambda'][idx_channel, :]
        if avg:
            data = np.mean(bepdata['spectra'][idx_channel, t0:t1, :], axis=0)
            if 'label' not in line_options:
                line_options['label'] = losname
            ax.plot(lambdaval, data, **line_options)
        else:
            data = bepdata['spectra'][idx_channel, t0:t1, :]
            for ii in range(data.shape[0]):
                if 'label' not in line_options:
                    line_options['label'] = '%s @ t = %.0f'%\
                        (losname, bepdata['time'][ii])
                else:
                    line_options['label'] += ' %s @ t = %.0f'%\
                        (losname, bepdata['time'][ii])
                ax.plot(lambdaval, data[ii, :], **line_options)

    if ax_was_none:
        ax_options['xlabel'] = 'Wavelength $\\lambda$ [nm]'
        ax_options['ylabel'] = 'Counts'

        ax = ssplt.axis_beauty(ax, ax_options)

        plt.tight_layout()

    return ax



# -----------------------------------------------------------------------------
# --- Functions reading and plotting the RAW BEP signals.
# -----------------------------------------------------------------------------
def bep_get_gain(emgain: int, adgain: int):
    """
    Transforms the pair of values (emgain, adgain) read from the BEP shotfile
    into the actual value of the gain and 'sad2'(?)

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param emgain: gain parameter from the BEP camera. Can only be 0 or 1.
    @param adgain: gain parameter from the BEP camera. Can be 1, 2 or 3.
    """

    if emgain == 0:
        if adgain == 1:
            gain = 0.487585
            sad2 = 3.30
        elif adgain == 2:
            gain = 1.00
            sad2 = 6.77
        elif adgain == 3:
            gain = 1.895
            sad2 = 12.83
        else:
            raise Exception('ADGAIN must be between 1 and 3, instead = %d!'%adgain)
    elif emgain >= 1:
        if adgain == 1:
            gain = 0.151
            sad2 = 4.0
        elif adgain == 2:
            gain = 0.255
            sad2 = 6.74
        elif adgain == 3:
            gain = 0.558
            sad2 = 14.77
        else:
            raise Exception('ADGAIN must be between 1 and 3, instead = %d!'\
                            %adgain)
    return gain, sad2**2.0


def BEPfromSF(shotnumber: int, time: float = None, edition: int=0):
    """
    Gets the raw BEP signal from the shotfile.
    No calibration nor corrections are applied.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: shotnumber of the shot to retrieve.
    @param timepoint: either time point or time window to get the BEP data.
    @param edition: edition of the shotfile.
    @return output: dictionary with all the parameters from SF and the data.
    """

    # --- Open the BEP and the NBI shotfiles.
    try:
        sf_bep = dd.shotfile(diagnostic='BEP', pulseNumber=shotnumber,
                             edition=edition, experiment='AUGD')
    except:
        raise Exception('Shotfile %05d for BEP cannot be opened!'%shotnumber)

    # --- Reading the parameters.
    # Number of pixels.
    npix = sf_bep.getParameter(setName='READ_CCD', parName='xdim').data

    # Number of channels = Number of Lines of Sight (LOS)
    nlos = sf_bep.getParameter(setName='READ_CCD', parName='ydim').data

    # Getting the timebase:
    timebase = sf_bep(name='TIME')
    if time is None:
        time = np.array((timebase[0], timebase[-1]))

    if np.mod(len(time), 2) != 0:
        time = np.append(time, timebase[-1])
    tidx = timebase.searchsorted(time)

    # --- Getting the LOS names
    losname  = dict()
    for ii in range(nlos):
        parName = 'CHAN_%02d'%(ii+1)
        losname[ii] = sf_bep.getParameter(setName='PARAM',
                             parName=parName).data.decode('utf8').strip()
    start_roi = sf_bep.getParameter(setName='READ_CCD', parName='starty').data
    end_roi   = sf_bep.getParameter(setName='READ_CCD', parName='endy').data
    len_roi   = sf_bep.getParameter(setName='READ_CCD', parName='groupy').data

    # --- Getting the parameters of the spectrometer:
    gratcons  = sf_bep.getParameter(setName='PARAM', parName='GRATCONS').data
    op_angle  = sf_bep.getParameter(setName='PARAM', parName='OP_ANG').data
    foc_len   = sf_bep.getParameter(setName='PARAM', parName='FOC_LEN').data
    pix_size  = sf_bep.getParameter(setName='PARAM', parName='PIXW').data
    pix_ord   = sf_bep.getParameter(setName='PARAM', parName='PIXORD').data
    setup_n   = sf_bep.getParameter(setName='PARAM', parName='SETUP').data
    camera_n  = sf_bep.getParameter(setName='PARAM', parName='CAMERA').data
    slit      = sf_bep.getParameter(setName='PARAM', parName='SLIT').data
    blende    = sf_bep.getParameter(setName='PARAM', parName='BLENDE_1').data
    emgain    = sf_bep.getParameter(setName='PARAM', parName='EM-GAIN').data
    adgain    = sf_bep.getParameter(setName='PARAM', parName='AD-GAIN').data

    gain, sad2 = bep_get_gain(emgain, adgain)

    cts_per_photon = FCTSPH*gain

    # --- Getting the wavelength from the motor.
    cwav = sf_bep.getParameter(setName='WL_MOTOR', parName='WAVE').data
    if shotnumber > 36900:
        cwav += 0.10

    # --- Reading the whole signal from the BEP shotfile.
    bepdata = sf_bep(name='CCD_DATA').data[tidx[0]:tidx[1], :]
    ntime = bepdata.shape[0]

    bepdata = np.reshape(bepdata, (ntime, nlos, npix))

    sf_bep.close()
    if camera_n == 1:
        losname[16] = 'neon'
    elif camera_n == 0:
        losname[15] = 'neon'

    # --- Setting up the output.
    output = { 'time': timebase[tidx[0]:tidx[1]],
               'conf': {'start_roi': start_roi,
                        'end_roi': end_roi,
                        'len_roi': len_roi,
                        'gratcons':gratcons,
                        'op_angle': op_angle,
                        'foc_len': foc_len,
                        'pix_size': pix_size,
                        'pix_ord': pix_ord,
                        'setup_n': setup_n,
                        'camera_n': camera_n,
                        'slit': slit,
                        'blende':blende,
                        'em-gain': emgain,
                        'ad-gain': adgain,
                        'gain': gain,
                        'sad2': sad2,
                        'cts_per_photon': cts_per_photon,
                        'cwav': cwav
                       },
               'nLOS': nlos,
               'nTime': ntime,
               'nPixels': npix,
               'LOSname': losname,
               'data': bepdata,
               'timeWindow': time,
               'shotnumber': shotnumber
             }

    return output


def plotBEP_fromSF(bepdata: dict, losname: str, ax = None,
                   verbose: bool = False, ax_options: dict={},
                   line_options:dict={}, time: float=None):
    """
    Plots into the axis provided (created if None is provided) the BEP raw
    signal associated to the LOS name.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param bepdata: dictionary with the data from the RAW shotfile.
    @param losname: name of the LOS to plot.
    @param verbose: write to console debug information.
    @param ax: axis to plot. If None, ones will be created.
    @param ax_options: dictionary with options for the axis.
    @param line_options: dictionary with options to plot.
    @param time: time window to plot. If None, all the input time window is
    used instead.
    """

    axis_was_none = False
    if ax is None:
        fig, ax = plt.subplots(1)
        axis_was_none = True

    los_found = False
    for ilos in bepdata['LOSname'].keys():
        if bepdata['LOSname'][ilos] == losname:
            los_found = True
            break

    if not los_found:
        raise Exception('LOS named %s not found!'%losname)

    if 'grid' not in ax_options:
        ax_options['grid'] = 'both'

    if 'fontsize' not in ax_options:
        ax_options['fontsize'] = 16

    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2

    if time is None:
        time = [bepdata['time'][0], bepdata['time'][-1]]

    t0, t1 = np.searchsorted(time)


    if verbose:
        ntime = t1 - t0
        dt    = bepdata['time'][1] - bepdata['time'][0]
        delta_time = ntime * dt
        print('Averaging over %d time-slices (%.3f ms)'%(ntime,
                                                         delta_time*1e3))

    plot_data = np.mean(bepdata['data'][t0:t1, ilos, :], axis=0)
    pix = np.arange(bepdata['data'].shape[-1])

    if 'label' not in line_options:
        line_options['label'] = '%05d - %s'%(bepdata['shotnumber'],
                                             losname)
    ax.plot(pix, plot_data, **line_options)


    if axis_was_none:
        ax_options['xlabel'] = 'Pixels'
        ax_options['ylabel'] = 'Non-corrected counts'

        ax = ssplt.axis_beauty(ax, ax_options)

        fig = plt.gcf()

        fig.tight_layout()
    return ax
