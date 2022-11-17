"""
Methods to include noise in the synthetic camera frame

Jose Rueda: jrrueda@us.es

In the future, it will include also routines to filter this noise

Introduced in version 0.4.13
"""
import numpy as np


def dark_plus_readout(frame, dark_centroid, sigma_readout):
    """
    Add dark plus readout noise

    Jose Rueda: jrrueda@us.es

    Notice: dark current and readout noise are effects always present, it is
    imposible to measure them independently, so they will be modelled as a
    single gaussian noise with centroid 'dark_centroid' and sigma
    'sigma_readout'. Both parameters to be measured for the used camera

    :param  frame: frame where we want to add the noise
    :param  dark_centroid: centroid for the noise level
    :param  sigma_readout: sigma of the noise
    """
    rand = np.random.default_rng()
    gauss = rand.standard_normal
    noise = dark_centroid + sigma_readout * gauss(frame.shape)
    return noise.astype(np.int)


def photon(frame, multiplier: float = 1.0):
    """
    Add photon noise

    Jose Rueda: jrrueda@us.es

    Noise will be constructed as a gaussian with sigma equal to sqrt(N), being
    N the number of counts at the pixel ij

    :param  frame: frame where we want to add the noise
    :param  multiplier: It serves as Fano factor, sigma = multiplier * sqrt(N)
    """
    rand = np.random.default_rng()
    gauss = rand.standard_normal
    noise = multiplier * np.sqrt(frame) * gauss(frame.shape)
    flags = noise < 0
    noise[flags] = 0
    return noise.astype(np.int)


def broken_pixels(frame, percent):
    """
    Simulate broken pixels

    Jose Rueda: jrrueda@us.es

    This will randomly cancel the signal in a number of pixels given by percent
    The idea is to simulate things like the broken fibers in the iHIBP

    :param  frame: Frame which we want to model the noise
    :param  percent: ratio of broken pixels, normalise to 1
    """
    rand = np.random.default_rng()
    uniform = rand.uniform(size=frame.shape)
    flags = uniform < percent
    dummy = frame.copy()
    if np.sum(flags) > 0:
        dummy[flags] = 0
    return dummy.astype(np.int)


def camera_neutrons(frame, percent=0.001, camera_bits=12, vmin=0.7,
                    vmax=1.0):
    """
    Add noise due to neutron impact on the sensor

    Jose Rueda: jrrueda@us.es

    :param  frame: frame where we want to add the noise
    :param  neutron_percent: percent (normalised to 1), of pixels affected
    :param  dtype:
    """
    rand = np.random.default_rng()
    uniform = rand.uniform(size=frame.shape)
    flags = uniform < percent
    dummy = frame.copy()
    if np.sum(flags) > 0:
        dummy2 = (2**camera_bits - 1)\
            * rand.uniform(low=vmin, high=vmax, size=frame.shape)[flags]
        dummy[flags] = dummy2.flatten()
    return dummy.astype(np.int)
