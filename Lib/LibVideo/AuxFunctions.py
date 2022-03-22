"""
Include auxiliar functions to load the video files.

Examples: rgb2gray or checking the timebase of the video

Jose Rueda: jrrueda@us.es
"""
import re
import os
import numpy as np


def rgb2gray(rgb):
    """
    Transform rgb images to gray.

    Jose Rueda: jrrueda@us.es

    It uses the scaling of MATLAB and opencv

    @param rgb: 3D matrix with the rgb information

    Note if the matrix include a transparency factor, 4 element of the RGB
    dimension, it will be ignored
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def check_timebase(timebase):
    """
    Check the time base looking for corrupted frames

    Jose Rueda: jrrueda@us.es

    Created because in some case, when we activate ICRH, CCD cameras of FILDs
    fails so for example up to t = 4.8 measured perfectly and them all
    time points become zero. This will detect if that happens

    @param timebase: time base of the frames (np.array)

    @return corrupt: flag to say if the timebase is corrupt or not
        - True: Is corrupted
        - False: Seems not corrupted
    """
    if np.sum(np.diff(timebase) < 0) > 0:
        corrupt = True
    else:
        corrupt = False
    return corrupt


def binary_image(frame, threshold, bool_flag: bool = True):
    """
    Set all pixels with signal larger than threshold to one and the rest to 0

    Jose Rueda. jrrueda@us.es

    @param frame: frame we want to process
    @param threshold: threshold for the algorithms
    @param bool_flag: if true the output will be bollean array
    @retun frame_new: 'masked frame' with just 0 and ones (int8)
    """
    frame_new = np.zeros(frame.shape, dtype='int8')
    frame_new[frame > threshold] = 1
    if bool_flag:
        return frame_new.astype(np.bool)
    else:
        return frame_new


def guess_shot(file, shot_number_length):
    """
    Guess the shot number from the name of the file

    Jose Rueda Rueda: jrrueda@us.es

    @param file: Name of the file or folder containing the data. In that
    name it is assumed to be the shot number in the proper format
    @param shot_number_length: Number of characters expected from the shot
    number in the file name (defined in the modulus of each machine)
    """
    list = re.findall(r'\d+', file)
    list = np.array(list)
    n = len(list)
    flags = np.zeros(n, dtype=np.bool)
    for i in range(n):
        if len(list[i]) == shot_number_length:
            flags[i] = True
    ntrues = np.sum(flags)
    if ntrues == 1:
        shot = int(list[flags])
    elif ntrues == 2:
        # Maybe just the file is saved in a folder named as the shot, so we
        # can have a second positive here
        options = list[flags]
        if options[0] == options[1]:
            shot = int(options[0])
    elif ntrues == 0:
        print('No shot number found in the name of the file')
        print('Give the shot number as input when loading the file')
        shot = None
    else:
        print('Several possibles shot number were found')
        print('Give the shot number as input when loading the file')
        print('Possible shot numbers ', list[flags])
        shot = None
    return shot
