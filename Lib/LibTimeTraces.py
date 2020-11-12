"""
Package to calculate time traces

Contains all the routines to calculate time traces. The RoiPoly module must be
installed. If you use Spyder as Python IDE, please install the RoiPoly version
which is compatible with Spyder. (See the readme of the project to get the link)

"""
import numpy as np
from roipoly import RoiPoly
import datetime
import time


def trace(frames, mask):
    """
    Calculate the trace of a region given by a mask

    Jose Rueda: ruejo@ipp.mpg.de

    Given the set of frames, frames[px,py,number_frames] and the mask[px,py]
    this function just sum all the pixels inside the mask for all the frames

    @param frames: 3D array containing the frames[px,py,number_frames]
    @param mask: Binary matrix which the pixels which must be consider,
    generated by roipoly
    @return tr: numpy array with the sum of the pixels inside the mask
    """
    # Allocate output array
    n = frames.shape[2]
    tr = np.zeros(n)
    # calculate the trace
    for iframe in range(n):
        tr[iframe] = sum(frames[mask, iframe])
    return tr


def create_roi(fig, re_display=False):
    """
    Wrapper for the RoiPoly features

    Jose Rueda: ruejo@ipp.mpg.de

    Just a wrapper for the roipoly capabilities which allows for the reopening
    of the figures
    @todo I can't understant why in spyder it does not work the .show....

    @param fig: fig object where the image is found
    @return fig: The figure with the roi plotted
    @return roi: The PloyRoi object
    """
    # Define the roi
    print('Please select the vertex of the roi in the figure')
    roi = RoiPoly(color='r', fig=fig)
    # Show again the image with the roi
    if re_display:
        fig.show()
        roi.display_roi()
    return fig, roi


def time_trace_cine(cin_object, mask, t1=0, t2=10):
    """
    Calculate the time trace from a cin file

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param cin_object: cin file object (see class cin of LibCinFiles.py)
    @param mask: binary mask defining the roi
    @param t1: initial time to calculate the timetrace [in s]
    @param t2: final time to calculate the timetrace [in s]
    @return tt: TimeTrace object
    """
    # --- Section 0: Initialise the arrays
    # Initialise the timetrace object
    tt = TimeTrace()
    tt.mask = mask

    # Look for the index in the data base
    i1 = (np.abs(cin_object.timebase - t1)).argmin()
    i2 = (np.abs(cin_object.timebase - t2)).argmin()
    # Initialise the arrays
    tt.time_base = cin_object.timebase[i1:i2]
    tt.sum_of_roi = np.zeros(i2 - i1)
    tt.mean_of_roi = np.zeros(i2 - i1)
    tt.std_of_roi = np.zeros(i2 - i1)
    # I could call for each time the read_cine_image, but that would imply to
    # open and close several time the file as well as navigate trough it... I
    # will create temporally a copy of that routine, this need to be
    # rewritten properly, but it should work and be really fast
    # ---  Section 1: Get frames position
    # Open file and go to the position of the image header
    print('Opening .cin file and reading data')
    tic = time.time()
    fid = open(cin_object.file, 'r')
    fid.seek(cin_object.header['OffImageOffsets'])
    # Position of the frames
    if cin_object.header['Version'] == 0:  # old format
        position_array = np.fromfile(fid, 'int32',
                                     int(cin_object.header['ImageCount']))
    else:
        position_array = np.fromfile(fid, 'int64',
                                     int(cin_object.header['ImageCount']))
    # 8-bits or 16 bits frames:
    size_info = cin_object.settings['RealBPP']
    if size_info <= 8:
        # BPP = 8  # Bits per pixel
        data_type = 'uint8'
    else:
        # BPP = 16  # Bits per pixel
        data_type = 'uint16'
    # Image size (in bytes)
    # img_size_header = int(cin_object.imageheader['biWidth'] *
    #                     cin_object.imageheader['biHeight']) * BPP / 8
    # Number of pixels
    npixels = cin_object.imageheader['biWidth'] * \
        cin_object.imageheader['biHeight']
    itt = 0  # Index to cover the array during the loop
    # --- Section 2: Read the frames
    for i in range(i1, i2):
        #  Go to the position of the file
        iframe = i  # - cin_object.header['FirstImageNo']
        fid.seek(position_array[iframe])
        #  Skip header of the frame
        length_annotation = np.fromfile(fid, 'uint32', 1)
        fid.seek(position_array[iframe] + length_annotation - 4)
        #  Read frame
        np.fromfile(fid, 'uint32', 1)  # Image size in bytes
        dummy = np.reshape(np.fromfile(fid, data_type,
                                       int(npixels)),
                           (int(cin_object.imageheader['biWidth']),
                            int(cin_object.imageheader['biHeight'])),
                           order='F').transpose()
        tt.sum_of_roi[itt] = np.sum(dummy[mask])
        tt.mean_of_roi[itt] = np.mean(dummy[mask])
        tt.std_of_roi[itt] = np.std(dummy[mask])
        itt = itt + 1
    fid.close()
    toc = time.time()
    print('Elapsed time [s]: ', toc - tic)
    return tt


class TimeTrace:
    """
    Class with information of the time trace
    """

    def __init__(self):
        """
        Initialise the TimeTrace
        """
        ## Numpy array with the time bas
        self.time_base = None
        ## Numpy array with the total counts in the ROI
        self.sum_of_roi = None
        ## Numpy array with the mean of counts in the ROI
        self.mean_of_roi = None
        ## Numpy array with the std of counts in the ROI
        self.std_of_roi = None
        ## Binary mask defining the roy
        self.mask = None
        ## Binary mask defining the roy
        self.roi = None

    def export_to_ascii(self, filename: str):
        """
        Export time trace to acsii

        Jose Rueda: jose.rueda@ipp.mpg.de

        @param self: the TimeTrace object
        @param filename: file where to write the data
        @return None. A file is created with the information
        """
        date = datetime.datetime.now()
        line = '# Time trace: ' + date.strftime("%d-%b-%Y (%H:%M:%S.%f)") \
               + '\n' + 'Time [s]                     ' + \
               'Counts in Roi                     ' + \
               'Mean in Roi                     Std Roi'
        length = self.time_base.size

        np.savetxt(filename, np.hstack((self.time_base.reshape(length, 1),
                                        self.sum_of_roi.reshape(length, 1),
                                        self.mean_of_roi.reshape(length, 1),
                                        self.std_of_roi.reshape(length, 1))),
                   delimiter='   ,   ', header=line)
        f = open("output.txt", "w")
        print("# Time trace ", file=f)
        print("# Date ", datetime.datetime.now(), file=f)
        f.close()
