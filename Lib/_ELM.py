"""
ELM analysis library.

This enables the study of ELM signals.


Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import Lib.LibData as libdata
import scipy
import warnings
from tqdm import tqdm


def getELM_times(shotnumber: int, time: float = None, diag: str = 'MAC',
                 signal: str = 'Ipolsola', exp: str = 'AUGD', edition: int = 0,
                 exp_elm: str = 'AUGD', **kwargs):
    """
    This routine will read from the AUG database the time traces of the ELMs
    as stored in the ELM shotfile. Then using a correlation approach,
    the actual starting points of the ELMs in the input signal.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: pulse number to study the ELM.
    :param  time: time window to analyze. If None, the whole time window
    is used.
    :param  diag: diagnostic to read the data to be analyzed.
    :param  signal: name of the signal to use.
    :param  exp: experiment where the shotfile is stored.
    :param  edition: edition to open.
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    try:
        elm = libdata.get_ELM_timebase(shot=shotnumber, time=time,
                                       exp=exp_elm)

    except:
        raise Exception('Cannot obtaint the ELM starting points for #%05d'\
                        % shotnumber)

    time, data = libdata.get_signal_generic(shot=shotnumber, diag=diag,
                                            signame=signal, exp=exp,
                                            edition=edition)

    data = np.abs(data)
    return getELM_times_from_signal_raw(elm_start=elm,
                                        time=time, signal=data, **kwargs)

def getELM_times_from_signal_raw(elm_start: dict, time: float, signal: float,
                                 buildCorr: bool = True,
                                 delay_start: float = 0.0,
                                 delay_end: float = 0.0):
    """
    This routine will read from the AUG database the time traces of the ELMs
    as stored in the ELM shotfile. Then using a correlation approach,
    the actual starting points of the ELMs in the input signal.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  elm_start: dictionary containing the data of the ELM onset and the
    ELM timing.
    :param  time: time basis of signal to get the ELM starting points and apply
    ELM synchr.
    :param  signal: signal for the ELM synchronization.
    :param  buildCorr: flag to compute the ELM correlation. Default to False
    (the correlation, although fast, implies some extra time).
    :param  delay_start:

    """

    # This set of starting and ending points for testing the correlations have
    # been found by hand for the Ipolsola signal, and seems to be good.
    dt_start = 5.0e-4
    dt_end   = 1.0e-3
    dt_test  = 1.0e-4

    t0_elm = np.searchsorted(time, elm_start['t_onset'][0])
    t1_elm = np.searchsorted(time, elm_start['t_onset'][0]+elm_start['dt'][0])

    t0 = np.searchsorted(time, elm_start['t_onset'][0]-dt_start)
    t0_minus = np.searchsorted(time, elm_start['t_onset'][0]-dt_start-dt_test)
    t0_plus = np.searchsorted(time, elm_start['t_onset'][0]-dt_start+dt_test)
    t1 = np.searchsorted(time, elm_start['t_onset'][0]+elm_start['dt'][0]+dt_end)
    # t1_minus = np.searchsorted(time, elm_start['t_onset'][0]\
    #                           # -dt_test+elm_start['dt'][0]+dt_end)
    t1_plus = np.searchsorted(time, elm_start['t_onset'][0]\
                              + dt_test + elm_start['dt'][0] + dt_end)

    # Let's look for the first ELM. We will do this using the correlation
    # between the real data within an interval of [-1ms, t_end+1ms] wrt the
    # time basis given, and a test function that emulates an ELM crash.

    dt1        = elm_start['dt'][0]
    corr_vec   = np.zeros((t0_plus-t0_minus+1))

    # fig, ax = plt.subplots(1)

    norm = signal[t0_minus:t1_plus].max()
    time_1stELM = time[t0_elm:t1_elm] - time[t0]
    x0   = time_1stELM[signal[t0_elm:t1_elm].argmax()]

    def fit_objective(x, tau, sigma, x0, A):
        return A*np.exp(x/tau)*np.exp(-(x-x0)**2.0/(2*(sigma**2.0)))

    p0 = (dt1, dt1/2.0, x0, norm)
    bounds = ([0, 0, x0*0.99, 0.9*norm], [np.inf, 10*dt1, 1.01*x0, 1.1*norm])
    popt, _ = scipy.optimize.curve_fit(fit_objective, time_1stELM,
                                       signal[t0_elm:t1_elm],
                                       p0=p0,
                                       bounds=bounds)

    # ax.plot(time_1stELM, fit_objective(time_1stELM, *popt), linewidth=3)

    for ii in range(t0_plus-t0_minus+1):
        t0_loop = t0 + ii
        t1_loop = t1 + ii
        timebasis1 = time[t0_loop:t1_loop] - time[t0]
        data1      = signal[t0_loop:t1_loop]
        test_function = fit_objective(timebasis1, *popt)

        corr_vec[ii], _ = scipy.stats.pearsonr(test_function, data1)
        del data1, test_function

    # Getting the point with the highest correlation factor:
    idx0 = corr_vec.argmax()

    # --- Creating the array with the ELM info.
    tstart_idx = np.zeros((elm_start['n']), dtype=int)
    tend_idx   = np.zeros((elm_start['n']), dtype=int)
    tstart_idx[0] = t0 + idx0

    dt_mean = np.maximum(elm_start['dt'][0], np.mean(elm_start['dt']))
    time0 = time[tstart_idx[0]]+dt_mean
    tend_idx[0] = np.searchsorted(time, (time0, ))

    dt_idx = tend_idx[0] - tstart_idx[0]

    # The first ELM is now used as reference.
    # xref = time[tstart_idx[0]:tend_idx[0]]
    yref = signal[tstart_idx[0]:tend_idx[0]]

    # --- Loop over the ELMs
    for ii in range(1, elm_start['n']):
        time0 = elm_start['t_onset'][ii]

        time0_start = time0 - dt_start
        time0_minus = time0_start - dt_test
        time0_plus = time0_start + dt_test
        time_array = (time0_start, time0_minus, time0_plus)

        t0, t0minus, t0plus = np.searchsorted(time, time_array)
        t1 = t0 + dt_idx

        ntrials = t0plus - t0minus + 1

        corr_vec = np.zeros((ntrials,))
        for kk in range(ntrials):
            t0_loop = t0 + kk
            t1_loop = t1 + kk
            timebasis1 = time[t0_loop:t1_loop] - time[t0]
            data1      = signal[t0_loop:t1_loop]

            corr_vec[kk], _ = scipy.stats.pearsonr(yref, data1)

        t0_shift = corr_vec.argmax()
        tstart_idx[ii] = t0_shift + t0

        tend_idx[ii] = t1 + t0_shift

        del t0
        del t1
        del t0_shift
        del time0
        del corr_vec

    # --- Obtaining ELM-correlation matrix.
    corr_matrix = np.zeros((elm_start['n'], elm_start['n']))

    if buildCorr:
        for ii in tqdm(np.arange(elm_start['n'])):
            data1 = signal[tstart_idx[ii]:tend_idx[ii]]
            for jj in range(elm_start['n']):
                data2 = signal[tstart_idx[jj]:tend_idx[jj]]

                corr_matrix[ii, jj], _ = scipy.stats.pearsonr(data1, data2)

    # Generating the output:
    delay_start_idx = int(delay_start/(time[1] - time[0]))
    delay_end_idx = int(delay_end/(time[1] - time[0]))

    tstart_idx += delay_start_idx
    tend_idx += delay_end_idx

    output = { 'source': { 'elm_start': elm_start,
                           'time': time,
                           'signal': signal
                         },

              'corrMatrix': corr_matrix,
              'tstart': tstart_idx,
              'tend': tend_idx,
              'tstart_val': time[tstart_idx],
              'tend_val': time[tend_idx],
              'time_0': time[tstart_idx[0]:tend_idx[0]] - \
                        elm_start['t_onset'][0]
             }

    # Adding to the output the cut signal containing only the ELMs.
    new_dt_idx = tend_idx[0]-tstart_idx[0]
    output['ELM'] = np.zeros((elm_start['n'], new_dt_idx))
    for ii in range(elm_start['n']):
        output['ELM'][ii, :] = signal[tstart_idx[ii]:tend_idx[ii]]

    return output


def ELM_buildCorrelation(ELMsignal: float, axis: int=0):
    """
    Builds the correlation matrix between the ELM signals along the provided
    axis.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  ELMsignal: float matrix with the ELM signal as a matrix with
    nELMs x time_base. Notice that the time basis must be the same!
    :param  axis: axis along the ELMs are located.
    """

    if axis != 0:
        ELMsignal = np.moveaxis(ELMsignal, (axis,), (0,))

    ELMsignal = np.atleast_2d(ELMsignal)

    nELMS = ELMsignal.shape[0]
    corr_matrix = np.zeros((nELMS, nELMS))

    for ii in range(nELMS):
        for jj in range(nELMS):
            corr_matrix[ii, jj], _ = scipy.stats.pearsonr(ELMsignal[ii, ...],
                                                          ELMsignal[jj, ...])

    return corr_matrix


def ELM_similarity(elm_dict: dict, threshold: float = 0.96, elm_idx: int=None):
    """
    From the ELM data and the correlation between them, this will choose the
    most similar ones, to be able to make an ELM average. If an ELM value is
    used as input, then, all the similar ELMs are taken.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  elm_dict: dictionary with the ELM entry (at least).
    :param  threshold: threshold to claim correlation between the ELMs and take
    them.
    :param  elm_idx: if different from None, all the ELMs in positive correlation
    with the given ELM index are taken. Otherwise, the pair of ELMs with the
    largest correlation is taken as initial feed for the algorithm.
    """

    # Checking that the ELM input is in the dictionary.
    if 'ELM' not in elm_dict:
        raise ValueError('The input dictionary must contain the ELM signals.')

    # If the correlation matrix is not in the data, compute it.
    if 'corrMatrix' not in elm_dict:
        elm_dict ['corrMatrix'] = ELM_buildCorrelation(elm_dict['ELM'])
    elif np.all(elm_dict['corrMatrix'] == 0):
        elm_dict ['corrMatrix'] = ELM_buildCorrelation(elm_dict['ELM'])

    # Checking the threshold value.
    if threshold <=0:
        raise ValueError('The threshold cannot be negative!')

    # If the ELM index is not provided, we find the pair with the largest
    # correlation. Temporarily, we set the diagonal value to 0.
    if elm_idx is None:
        np.fill_diagonal(elm_dict['corrMatrix'],  0.0)

        idx_max = np.unravel_index(elm_dict['corrMatrix'].argmax(),
                                   elm_dict['corrMatrix'].shape)

        elm_idx = idx_max[0]  # The matrix is symmetric, it does not matter.
        np.fill_diagonal(elm_dict['corrMatrix'],  1.0)

    # Getting the row of the correlation matrix corresponding to elm_idx
    corr_vec = elm_dict['corrMatrix'][elm_idx, :]

    # Getting all the ELMs correlated with the ELM in the index.
    flags = corr_vec >= threshold
    data_elm = elm_dict['ELM'][flags, :]
    time_start = elm_dict['tstart_val'][flags]
    time_end = elm_dict['tend_val'][flags]
    t_onset  = elm_dict['source']['elm_start']['t_onset'][flags]

    # --- Computing the ELM average.
    avg_elm = np.mean(data_elm, axis=0)
    std_elm = np.std(data_elm, axis=0)

    # Generating the time basis.

    output = { 'time': elm_dict['time_0'],
               'ELM': data_elm,
               'avg': avg_elm,
               'std': std_elm,
               'nELM': data_elm.shape[0],
               'tstart_val': time_start,
               'tend_val': time_end,
               't_onset': t_onset
             }

    return output


def ELMsync(time: float, signal: float, elm_dict:dict, average = False):
    """
    Using the time synchonization provided by the elm_dict, the (time, signal)
    pair is parsed to return only the ELM synced time and data and be able
    to make ELM-averaging.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  time: new time basis to analyze.
    :param  signal: signal to apply the ELM synchronization. The time axis
    must be the first axis. This enables multiple signals to be synced
    simultaneously.
    :param  elm_dict: dictionary with the ELM data (output of ELM_similarity.)
    """

    time   = np.atleast_1d(time)
    signal = np.atleast_1d(signal)

    if 'ELM' not in elm_dict:
        raise KeyError('The key ELM must be in the elm_dict input!')

    if 'tstart_val' not in elm_dict:
        raise KeyError('The key tstart_val must be in the elm_dict input!')
    if 'tend_val' not in elm_dict:
        raise KeyError('The key tend_val must be in the elm_dict input!')

    new_time = list()
    new_signal = list()

    # this is needed in case we want to average the ELM windows #AJVV
    t0, t1 = np.searchsorted(time, (elm_dict['tstart_val'][0],
                                elm_dict['tend_val'][0]))
    t_length = t1-t0

    for ii in range(len(elm_dict['tstart_val'])):
        t0, t1 = np.searchsorted(time, (elm_dict['tstart_val'][ii],
                                        elm_dict['tend_val'][ii]))

        if t0 == t1:
            continue

        if not average:
            new_time.append(time[t0:t1]-elm_dict['t_onset'][ii])
            new_signal.append(signal[t0:t1, ...])
        else:
            if t1-t0 == t_length:
                '''
                only append arrays of ELM windows with equal length,
                otherwise np.mean will not work later.
                Sometimes the ELM window arrays differ in length by one index
                '''
                new_time.append(time[t0:t1]-elm_dict['t_onset'][ii])
                new_signal.append(signal[t0:t1, ...])

    if average:
        new_time   = np.array(new_time)
        new_signal = np.array(new_signal)

        new_time = np.mean(new_time, axis = 0)
        new_signal = np.mean(new_signal, axis = 0)

    else:
        new_time = np.concatenate(new_time)#.squeeze()
        new_signal = np.concatenate(new_signal)#.squeeze()

    return new_time, new_signal
