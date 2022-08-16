"""
Routines to read pco (.b16) files.

Written by Jose Rueda: jrrueda@us.es
Adapted by Lina Velarde: lvelarde@us.es
"""
import os
import logging
import numpy as np
logger = logging.getLogger('ScintSuite.Video')
try:
    from pco_tools import pco_reader as pco
except ModuleNotFoundError:
    logger.warning('0 :PCO tools not imported. No support for pco files')


def read_data(path: str, adfreq: float, t_trig: float):
    """
    Read info for a case where the measurements are stored as .b16

    Jose Rueda Rueda: jrrueda@us.es
    Lina Velarde: lvelarde@us.es

    Return a series of dictionaries similar to the case of a cin file,
    with all the info we can extract from the pco files

    @param path: path to the folder where the pngs are located
    @return time_base: time base of the frames (s)
    @return image_header: dictionary containing the info about the image size,
    and shape
    @return header: Header 'similar' to the case of a cin file
    @return settings: dictionary similar to the case of the cin file (it only
    contains the exposition time)
    """
    # Look for a png to extract the file and a .txt for the time information
    # f = []
    look_for_pco = True
    counter = 0
    for file in os.listdir(path):
        # if file.endswith('.txt'):
        #     f.append(os.path.join(path, file))
        if file.endswith('.b16'):
            counter += 1
            if look_for_pco:
                dummy = pco.load(os.path.join(path, file))
                si = dummy.shape
                imageheader = {
                    'biWidth': si[0],
                    'biHeight': si[1],
                    'framesDtype': dummy.dtype}
                look_for_pco = False
    # If no png was found, raise and error
    header = {'ImageCount': counter}
    if look_for_pco:
        print('No .b16 files in the folder...')
        return 0, 0, 0, 0
    # Possible bytes per pixels for the camera
    BPP = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}
    settings = {}
    try:
        settings['RealBPP'] = BPP[imageheader['framesDtype'].name]
        text = 'In the PCO there is no info about the real BitesPerPixel'\
            + ' used in the camera. Assumed that the BPP coincides with'\
            + ' the byte size of the variable!!!'
        print(text)
    except KeyError:
        raise Exception('Expected uint8,16,32,64 in the frames')
    # Generation of time_base
    time_base = np.arange(counter, dtype=float)/adfreq + t_trig

    return header, imageheader, settings, time_base[:].flatten()


def read_frame(video_object, frames_number=None, limitation: bool = True,
               limit: int = 2048, verbose: bool = True):
    """
    Read .b16 files

    Jose Rueda: jrrueda@us.es
    Lina Velarde: lvelarde@us.es

    @param video_object: Video class with the info of the video. See the video
        object of the BasicVideoObject.py file
    @param frames_number: array with the number of the frames to be loaded,
    if none, all frames will be loaded
    @param limitation: if we want to set a limitation of the size we can load
    @param limit: Limit to the size, in megabytes

    @return M: array of frames, [px in x, px in y, number of frames]
    """
    # Frames would have a name as CCD_qe_frame.b16 example: CCD_qe_0001.b16
    print('Reading PCO files')
    # check the size of the files, data will be saved as float32
    size_frame = video_object.imageheader['biWidth'] * \
        video_object.imageheader['biWidth'] * 2 / 1024 / 1024
    if frames_number is None:
        # In this case, we load everything
        if limitation and \
                size_frame * video_object.header['ImageCount'] > limit:
            raise Exception('Loading all frames is too much')
            return 0

        M = np.zeros((video_object.imageheader['biWidth'],
                      video_object.imageheader['biHeight'],
                      video_object.header['ImageCount']),
                     dtype=video_object.imageheader['framesDtype'])
        counter = 0
        for file in sorted(os.listdir(video_object.path)):
            if file.endswith('.b16'):
                M[:, :, counter] = pco.load(
                    os.path.join(video_object.path, file))
                # print(file)
                counter = counter + 1
    else:
        # Load only the selected frames
        counter = 0
        current_frame = 0
        if limitation and \
                size_frame * len(frames_number) > limit:
            print('Loading all frames is too much')
            return 0
        M = np.zeros((video_object.imageheader['biWidth'],
                      video_object.imageheader['biHeight'],
                      len(frames_number)),
                     dtype=video_object.imageheader['framesDtype'])

        for file in sorted(os.listdir(video_object.path)):
            if file.endswith('.b16'):
                current_frame = current_frame + 1
                if current_frame in frames_number:
                    pngname = os.path.join(video_object.path, file)
                    dummy = pco.load(pngname)
                    M[:, :, counter] = dummy
                    counter = counter + 1
        print('Number of loaded frames: ', counter)
    return M
