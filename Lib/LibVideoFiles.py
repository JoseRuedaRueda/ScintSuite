"""Handle the video from the different cameras

This module is created to handle the .cin (.cine) files, binary files
created by the Phantom cameras. In its actual state it can read everything
from the file, but it can't write/create a cin file. It also load data from
PNG files as the old FILD_GUI and will be able to work with tiff files
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import LibMap as ssmap
import LibPaths as p
from matplotlib.widgets import Slider
from LibMachine import machine
from version_suite import version
from scipy.io import netcdf                # To export remap data
from skimage import io                     # To load png images
import tkinter as tk                       # To open UI windows
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
pa = p.Path(machine)
del p
if machine == 'AUG':
    import LibDataAUG as ssdat
try:
    import cv2
except ModuleNotFoundError:
    print('There would be no support for the mp4 videos, open cv not found')


# -----------------------------------------------------------------------------
# --- Methods for the .cin files
# -----------------------------------------------------------------------------
def read_header(filename: str, verbose: bool = False):
    """
    Read the header info of a .cin file.

    Jose Rueda: ruejo@ipp.mpg.de

    @param filename: name of the file to open (path to file, not just the name)
    @param verbose: Optional, to display the content of the header
    @return cin_header: dictionary containing header information, see the
    inline comments in the code for a full description of each dictionary field
    """

    # Section 1: Read the file header
    fid = open(filename, 'r')
    if verbose:
        print('Reading .cin header')

    # Sign number of all.cin files
    cin_header = {'Type': np.fromfile(fid, 'S2', 1),
                  'Headersize': np.fromfile(fid, 'uint16', 1),
                  'Compression': np.fromfile(fid, 'uint16', 1)}
    # It represent the cine file header structure size as number of bytes
    # Compression applied to the data
    if cin_header['Compression'] == 0:
        if verbose:
            print('The video is saved as grey images\n')
    elif cin_header['Compression'] == 1:
        if verbose:
            print('The video is saved as jpeg compressed files \n')
    elif cin_header['Compression'] == 2:
        if verbose:
            print('The video is saved as raw uninterpolated color\n')
    else:
        print('Compression : ', cin_header['Compression'])
        print('Error reading the header, compression number wrong')
        return cin_header
    # Version of the cine file
    cin_header['Version'] = np.fromfile(fid, 'uint16', 1)
    if cin_header['Version'] == 0:
        print('This is a rather old version of the cine file, there could be '
              'some problems, stay alert\n')
    # number of the first recorded frame, relative to the trigger event
    cin_header['FirstMovieImage'] = np.fromfile(fid, 'int32', 1)
    # total number of frames taken by the camera
    cin_header['TotalImageCount'] = np.fromfile(fid, 'uint32', 1)
    # number of the first saved frame in the .cin file, relative to the trigger
    cin_header['FirstImageNo'] = np.fromfile(fid, 'int32', 1)
    # number of saved frames
    cin_header['ImageCount'] = np.fromfile(fid, 'uint32', 1)
    # Offset of the BITMAPINFOHEADER structure in the cine file
    cin_header['OffImageHeader'] = np.fromfile(fid, 'uint32', 1)
    # Offset of the SETUP structure in the cine file
    cin_header['OffSetup'] = np.fromfile(fid, 'uint32', 1)
    # Offset in the cine file of an array with the positions of each image
    # stored in the file
    cin_header['OffImageOffsets'] = np.fromfile(fid, 'uint32', 1)
    # Trigger time
    cin_header['TriggerTime'] = {'fractions': np.fromfile(fid, 'uint32', 1),
                                 'seconds': np.fromfile(fid, 'uint32', 1)}
    # Close the file
    fid.close()

    # Print if needed
    if verbose:
        for y in cin_header:
            print(y, ':', cin_header[y])

    # Return
    return cin_header


def read_settings(filename: str, bit_pos: int, verbose: bool = False):
    """
    Read the settings from a .cin file

    Jose Rueda: jrrueda@us.es

    @param filename: name of the file to read (full path)
    @param bit_pos: Position of the file where the setting structure starts
    @param verbose: verbose results or not
    @return cin_settings: dictionary containing all the camera settings. See
    inline comments at the end of the function
    @todo Check if the coefficient matrix of UF is ok or must be transposed
    """

    # Open file and go to the position of the settins structure
    fid = open(filename, 'r')
    if verbose:
        print('Reading .cin settings')
    fid.seek(bit_pos)

    # Requested frame rate
    cin_settings = {'FrameRate16': np.fromfile(fid, 'uint16', 1),
                    'Shutter16': np.fromfile(fid, 'uint16', 1),
                    'PostTrigger16': np.fromfile(fid, 'uint16', 1),
                    'FrameDelay16': np.fromfile(fid, 'uint16', 1),
                    'AspectRatio': np.fromfile(fid, 'uint16', 1),
                    'Res7': np.fromfile(fid, 'uint16', 1),
                    'Res8': np.fromfile(fid, 'uint16', 1),
                    'Res9': np.fromfile(fid, 'uint8', 1),
                    'Res10': np.fromfile(fid, 'uint8', 1),
                    'Res11': np.fromfile(fid, 'uint8', 1),
                    'TrigFrame': np.fromfile(fid, 'uint8', 1),
                    'Res12': np.fromfile(fid, 'uint8', 1),
                    'DescriptionOld': np.fromfile(fid, 'S121', 1),
                    'Mark': np.fromfile(fid, 'uint16', 1),
                    'Length': np.fromfile(fid, 'uint16', 1),
                    'Res13': np.fromfile(fid, 'uint16', 1),
                    'SigOption': np.fromfile(fid, 'uint16', 1),
                    'BinChannels': np.fromfile(fid, 'uint16', 1),
                    'SamplesPerImage': np.fromfile(fid, 'uint8', 1),
                    'BinName': np.fromfile(fid, 'S11', 8),
                    'AnaOption': np.fromfile(fid, 'uint16', 1),
                    'AnaChannels': np.fromfile(fid, 'uint16', 1),
                    'Res6': np.fromfile(fid, 'uint8', 1),
                    'AnaBoard': np.fromfile(fid, 'uint8', 1),
                    'ChOption': np.fromfile(fid, 'int16', 8),
                    'AnaGain': np.fromfile(fid, 'float32', 8),
                    'AnaUnit': np.fromfile(fid, 'S6', 8),
                    'AnaName': np.fromfile(fid, 'S11', 8),
                    'lFirstImage': np.fromfile(fid, 'int32', 1),
                    'dwImageCount': np.fromfile(fid, 'uint32', 1),
                    'nQFactor': np.fromfile(fid, 'int16', 1),
                    'wCineFileType': np.fromfile(fid, 'uint16', 1),
                    'szCinePath': np.fromfile(fid, 'S65', 4),
                    'Res14': np.fromfile(fid, 'uint16', 1),
                    'Res15': np.fromfile(fid, 'uint8', 1),
                    'Res16': np.fromfile(fid, 'uint8', 1),
                    'Res17': np.fromfile(fid, 'uint16', 1),
                    'Res18': np.fromfile(fid, 'float64', 1),
                    'Res19': np.fromfile(fid, 'float64', 1),
                    'Res20': np.fromfile(fid, 'uint16', 1),
                    'Res1': np.fromfile(fid, 'int32', 1),
                    'Res2': np.fromfile(fid, 'int32', 1),
                    'Res3': np.fromfile(fid, 'int32', 1),
                    'ImWidth': np.fromfile(fid, 'uint16', 1),
                    'ImHeight': np.fromfile(fid, 'uint16', 1),
                    'EDRShutter16': np.fromfile(fid, 'uint16', 1),
                    'Serial': np.fromfile(fid, 'uint32', 1)}
    if cin_settings['Serial'] == 24167 and verbose:
        print('Data was saved with a camera belonging to Prof. E Viezzer')
    cin_settings['Saturation'] = np.fromfile(fid, 'int32', 1)
    cin_settings['Res5'] = np.fromfile(fid, 'uint8', 1)
    cin_settings['AutoExposure'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['bFlipH'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['bFlipV'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['Grid'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FrameRate'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['Shutter'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['EDRShutter'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['PostTrigger'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FrameDelay'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['bEnableColor'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['CameraVersion'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FirmwareVersion'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['SoftwareVersion'] = np.fromfile(fid, 'uint32', 1)
    if cin_settings['SoftwareVersion'] <= 551:
        return cin_settings
    cin_settings['RecordingTimeZone'] = np.fromfile(fid, 'int32', 1)
    if cin_settings['SoftwareVersion'] <= 552:
        return cin_settings
    cin_settings['CFA'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['Bright'] = np.fromfile(fid, 'int32', 1)
    cin_settings['Contrast'] = np.fromfile(fid, 'int32', 1)
    cin_settings['Gamma'] = np.fromfile(fid, 'int32', 1)
    cin_settings['Res21'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['AutoExpLevel'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['AutoExpSpeed'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['AutoExpRect'] = {'left': np.fromfile(fid, 'uint32', 1),
                                   'top': np.fromfile(fid, 'uint32', 1),
                                   'right': np.fromfile(fid, 'uint32', 1),
                                   'bottom': np.fromfile(fid, 'uint32', 1)}
    cin_settings['WBGain'] = np.fromfile(fid, 'float32', 8)
    cin_settings['Rotate'] = np.fromfile(fid, 'int32', 1)
    cin_settings['WBView'] = np.fromfile(fid, 'float32', 2)
    cin_settings['RealBPP'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['Conv8Min'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['Conv8Max'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FilterCode'] = np.fromfile(fid, 'int32', 1)
    cin_settings['FilterParam'] = np.fromfile(fid, 'int32', 1)
    cin_settings['UF'] = {'dim': np.fromfile(fid, 'int32', 1),
                          'shifts': np.fromfile(fid, 'int32', 1),
                          'bias': np.fromfile(fid, 'int32', 1),
                          'Coef': np.fromfile(fid, 'int32', 25).reshape((
                                                                        5, 5))}
    cin_settings['BlackCalSVer'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['WhiteCalSVer'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['GrayCalSVer'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['bStampTime'] = bool(np.fromfile(fid, 'int32', 1))
    if cin_settings['SoftwareVersion'] <= 605:
        return cin_settings
    cin_settings['SoundDest'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FRPSteps'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FRPImgNr'] = np.fromfile(fid, 'int32', 16)
    cin_settings['FRPRate'] = np.fromfile(fid, 'uint32', 16)
    cin_settings['FRPExp'] = np.fromfile(fid, 'uint32', 16)
    cin_settings['MCCnt'] = np.fromfile(fid, 'int32', 1)
    cin_settings['MCPercent'] = np.fromfile(fid, 'float32', 64)
    if cin_settings['SoftwareVersion'] <= 606:
        return cin_settings
    cin_settings['CICalib'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CalibWidth'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CalibHeight'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CalibRate'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CalibExp'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CalibEDR'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CalibTemp'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['HeadSerial'] = np.fromfile(fid, 'uint32', 4)
    if cin_settings['SoftwareVersion'] <= 607:
        return cin_settings
    cin_settings['RangeCode'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['RangeSize'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['Decimation'] = np.fromfile(fid, 'uint32', 1)
    if cin_settings['SoftwareVersion'] <= 614:
        return cin_settings
    cin_settings['MasterSerial'] = np.fromfile(fid, 'uint32', 1)
    if cin_settings['SoftwareVersion'] <= 624:
        return cin_settings
    cin_settings['Sensor'] = np.fromfile(fid, 'uint32', 1)
    if cin_settings['SoftwareVersion'] <= 625:
        return cin_settings
    # --- Adquisition parameters in NS
    cin_settings['ShutterNs'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['EDRShutterNs'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['FrameDelayNs'] = np.fromfile(fid, 'uint32', 1)
    if cin_settings['SoftwareVersion'] <= 631:
        return cin_settings
    cin_settings['ImPosXAcq'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['ImPosYAcq'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['ImWidthAcq'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['ImHeightAcq'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['Description'] = np.fromfile(fid, 'S4096', 1)
    if cin_settings['SoftwareVersion'] <= 637:
        return cin_settings
    cin_settings['RisingEdge'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['FilterTime'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['LongReady'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['ShutterOff'] = bool(np.fromfile(fid, 'int32', 1))
    if cin_settings['SoftwareVersion'] <= 658:
        return cin_settings
    cin_settings['Res4'] = np.fromfile(fid, 'uint8', 16)
    if cin_settings['SoftwareVersion'] <= 663:
        return cin_settings
    cin_settings['bMetaWB'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['Hue'] = np.fromfile(fid, 'int32', 1)
    if cin_settings['SoftwareVersion'] <= 671:
        return cin_settings
    cin_settings['BlackLevel'] = np.fromfile(fid, 'int32', 1)
    cin_settings['WhiteLevel'] = np.fromfile(fid, 'int32', 1)
    # --- Lenses descriptions
    cin_settings['LensDescription'] = np.fromfile(fid, 'S256', 1)
    cin_settings['LensAperture'] = np.fromfile(fid, 'float32', 1)
    cin_settings['LensFocusDistance'] = np.fromfile(fid, 'float32', 1)
    cin_settings['LensFocalLength'] = np.fromfile(fid, 'float32', 1)
    if cin_settings['SoftwareVersion'] <= 691:
        return cin_settings

    # --- Image Adjustement
    cin_settings['fOffset'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fGain'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fSaturation'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fHue'] = np.fromfile(fid, 'float32', 1)
    cin_settings['Gamma'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fGammaR'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fGammaB'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fFlare'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fPedestalR'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fPedestalG'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fPedestalB'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fChroma'] = np.fromfile(fid, 'float32', 1)

    cin_settings['ToneLabel'] = np.fromfile(fid, 'S256', 1)
    cin_settings['TonePoints'] = np.fromfile(fid, 'int32', 1)
    cin_settings['fTone'] = np.fromfile(fid, 'float32', 64)

    cin_settings['UserMatrixLabel'] = np.fromfile(fid, 'S256', 1)

    cin_settings['EnableMatrices'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['cmUser'] = np.fromfile(fid, 'float32', 9)

    cin_settings['EnableCrop'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['CropRect'] = {'left': np.fromfile(fid, 'int32', 1),
                                'top': np.fromfile(fid, 'int32', 1),
                                'right': np.fromfile(fid, 'int32', 1),
                                'bottom': np.fromfile(fid, 'int32', 1)}

    cin_settings['EnableResample'] = bool(np.fromfile(fid, 'int32', 1))
    cin_settings['ResampleWidth'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['ResampleHeight'] = np.fromfile(fid, 'uint32', 1)

    cin_settings['fGain16_8'] = np.fromfile(fid, 'float32', 1)
    if cin_settings['SoftwareVersion'] <= 693:
        return cin_settings
    cin_settings['FRPShape'] = np.fromfile(fid, 'int32', 16)
    # This is temporal!!! I have to learn how to read single bits in python
    cin_settings['TrigTC'] = np.fromfile(fid, 'uint8', 8)
    cin_settings['fPbRate'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fTcRate'] = np.fromfile(fid, 'float32', 1)
    if cin_settings['SoftwareVersion'] <= 701:
        return cin_settings
    cin_settings['CineName'] = np.fromfile(fid, 'S256', 1)
    if cin_settings['SoftwareVersion'] <= 702:
        return cin_settings
    cin_settings['fGainR'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fGainG'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fGainB'] = np.fromfile(fid, 'float32', 1)
    cin_settings['cmCalib'] = np.fromfile(fid, 'float32', 9)
    cin_settings['fWBTemp'] = np.fromfile(fid, 'float32', 1)
    cin_settings['fWBCc'] = np.fromfile(fid, 'float32', 1)
    cin_settings['CalibrationInfo'] = np.fromfile(fid, 'S1024', 1)
    cin_settings['OpticalFilter'] = np.fromfile(fid, 'S1024', 1)
    if cin_settings['SoftwareVersion'] <= 709:
        return cin_settings
    cin_settings['GpsInfo'] = np.fromfile(fid, 'S256', 1)
    cin_settings['Uuid'] = np.fromfile(fid, 'S256', 1)
    if cin_settings['SoftwareVersion'] <= 719:
        return cin_settings
    cin_settings['CreatedBy'] = np.fromfile(fid, 'S256', 1)
    if cin_settings['SoftwareVersion'] <= 720:
        return cin_settings
    cin_settings['RecBPP'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['LowestFormatBPP'] = np.fromfile(fid, 'uint16', 1)
    cin_settings['LowestFormatQ'] = np.fromfile(fid, 'uint16', 1)
    if cin_settings['SoftwareVersion'] <= 731:
        return cin_settings
    cin_settings['fToe'] = np.fromfile(fid, 'float32', 1)
    cin_settings['LogMode'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CameraModel'] = np.fromfile(fid, 'S256', 1)
    if cin_settings['SoftwareVersion'] <= 742:
        return cin_settings
    cin_settings['WBType'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['fDecimation'] = np.fromfile(fid, 'float32', 1)
    if cin_settings['SoftwareVersion'] <= 745:
        return cin_settings
    cin_settings['MagSerial'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['CSSerial'] = np.fromfile(fid, 'uint32', 1)
    cin_settings['dFrameRate'] = np.fromfile(fid, 'float64', 1)
    if cin_settings['SoftwareVersion'] <= 751:
        return cin_settings
    cin_settings['SensorMode'] = np.fromfile(fid, 'uint32', 1)
    if cin_settings['SoftwareVersion'] <= 771:
        return cin_settings
    fid.close()
    # --- return and print
    # Print if needed
    if verbose:
        for y in cin_settings:
            print(y, ':', cin_settings[y])
    #  uint16_t FrameRate16;     // Frame rate fps---UPDF replaced by FrameRate
    #  uint16_t PostTrigger16;   // ---UPDF replaced by PostTrigger
    #  uint16_t FrameDelay16;    // ---UPDF replaced by FrameDelayNs
    #  uint16_t AspectRatio;     // ---UPDF replaced by ImWidth, ImHeight
    #  uint16_t Res7;            // ---TBI Contrast16
    #                            // (analog controls, not available after
    #                            // Phantom v3)
    #  uint16_t Res8;            // ---TBI Bright16
    #  uint8_t Res9;             // ---TBI Rotate16
    #  uint8_t Res10;            // ---TBI TimeAnnotation
    #                            // (time always comes from camera )
    #  uint8_t Res11;            // ---TBI TrigCine (all cines are triggered)
    #  uint8_t TrigFrame;        // Sync imaging mode:
    #                            // 0, 1, 2 = internal, external, locktoirig
    #  uint8_t Res12;            // ---TBI ShutterOn (the shutter is always on)
    #  char DescriptionOld[MAXLENDESCRIPTION_OLD];
    #                         // ---UPDF replaced by larger Description able to
    #                            // store 4k of user comments
    #  uint16_t Mark;            // "ST" - marker for setup file
    #  uint16_t Length;          // Length of the current version of setup
    #  uint16_t Res13;           // ---TBI Binning (binning factor)
    #  uint16_t SigOption;       // Global signals options:
    #                          // MAXSAMPLES = records the max possible samples
    #  int16_t BinChannels;      // Number of binary channels read from the
    #                            // SAM (Signal Acquisition Module)
    #  uint8_t SamplesPerImage;  // Number of samples acquired per image, both
    #                            // binary and analog;
    #  char BinName[8][11];      // Names for the first 8 binary signals having
    #                           //maximum 10 chars/name; each string ended by a
    #                            // byte = 0
    #  uint16_t AnaOption;       // Global analog options single ended, bipolar
    #  int16_t AnaChannels;      // Number of analog channels used (16 bit 2's
    #                            // complement per channel)
    #  uint8_t Res6;             // ---TBI (reserved)
    #  uint8_t AnaBoard;         // Board type 0=none, 1=dsk (DSP system kit),
    #                            // 2 dsk+SAM
    #                            // 3 Data Translation DT9802
    #                            // 4 Data Translation DT3010
    #  int16_t ChOption[8];      // Per channel analog options;
    #                            // now:bit 0...3 analog gain (1,2,4,8)
    #  float AnaGain[8];    // User gain correction for conversion from voltage
    #                            // to real units , per channel
    #  char AnaUnit[8][6];       // Measurement unit for analog channels: max 5
    #                            // chars/name ended each by a byte = 0
    #  char AnaName[8][11];    // Channel name for the first 8 analog channels:
    #                            // max 10 chars/name ended each by a byte = 0
    #  int32_t lFirstImage;      // Range of images for continuous recording:
    #                            // first image
    #  uint32_t dwImageCount;    // Image count for continuous recording;
    #                            // used also for signal recording
    #  int16_t nQFactor;         // Quality - for saving to compressed file at
    #                            // continuous recording; range 2...255
    #  uint16_t wCineFileType;   // Cine file type - for continuous recording
    #  char szCinePath[4][OLDMAXFILENAME]; //4 paths to save cine files - for
    #                         // continuous recording. After upgrading to Win32
    #                            // this still remained 65 bytes long each
    #                            // GetShortPathName is used for the filenames
    #                            // saved here
    #  uint16_t Res14;           // ---TBI bMainsFreq (Mains frequency:
    #                            // TRUE = 60Hz USA, FALSE = 50Hz
    #                            // Europe, for signal view in DSP)
    #                            // Time board - settings for PC104 irig board
    #                           // used in Phantom v3 not used anymore after v3
    #  uint8_t Res15;            // ---TBI bTimeCode;
    #                            // Time code: IRIG_B, NASA36, IRIG-A
    #  uint8_t Res16;            // --TBI bPriority
    #                            // Time code has priority over PPS
    #  uint16_t Res17;           // ---TBI wLeapSecDY
    #                            // Next day of year with leap second
    #  double Res18;         // ---TBI dDelayTC Propagation delay for time code
    #  double Res19;         // ---TBI dDelayTC Propagation delay for time code
    #  uint16_t Res20;           // ---TBI GenBits
    #  int32_t Res1;             // ---TBI GenBits
    #  int32_t Res2;             // ---TBI GenBits
    #  int32_t Res3;             // ---TBI GenBits
    #  uint16_t ImWidth;     // Image dimensions in v4 and newer cameras: Width
    #  uint16_t ImHeight;        // Image height
    #  uint16_t EDRShutter16;    // ---UPDF replaced by EDRShutterNs
    #  uint32_t Serial;          // Camera serial number
    #  int32_t Saturation;       // ---UPDF replaced by float fSaturation
    #                      // Color saturation adjustmment [-100,100] neutral 0
    #  uint8_t Res5;             // --- TBI
    #  uint32_t AutoExposure// Autoexposure enable 0=disable, 1=lock at trigger
    #                            // 3=active after trigger
    #  bool32_t bFlipH;          // Flips image horizontally
    #  bool32_t bFlipV;          // Flips image vertically
    #  uint32_t Grid;     // Displays a crosshair or a grid in setup, 0=no grid
    #                            // 2=cross hair, 8= grid with 8 intervals
    #  uint32_t FrameRateInt;    // Frame rate in frames per seconds
    #  uint32_t Shutter;         // ---UPDF replaced by ShutterNs
    #                            // (here the value is in microseconds)
    #  uint32_t EDRShutter;      // ---UPDF replaced by EDRShutterNs
    #                            // (here the value is in microseconds)
    #  uint32_t PostTrigger;     // Post trigger frames, measured in frames
    #  uint32_t FrameDelay;      // ---UPDF replaced by FrameDelayNs
    #                            // (here the value is in microseconds)
    #  bool32_t bEnableColor;    // User option: when 0 forces gray images from
    #                            // color cameras
    #  uint32_t CameraVersion;//The version of camera hardware (without decimal
    #                         // point). Examples of cameras produced after the
    #                            // year 2000
    #                            // Firewire: 4, 5, 6
    #                            // Ethernet: 42 43 51 7 72 73 9 91 10
    #                            // 650 (p65) 660 (hd) ....
    #  uint32_t FirmwareVersion; // Firmware version
    #  uint32_t SoftwareVersion; // Phantom software version
    #                        // End of SETUP in software version 551 (May 2001)
    #  int32_t RecordingTimeZone;//The time zone active during the recording of
    #                            // the cine
    #                        // End of SETUP in software version 552 (May 2001)
    #  uint32_t CFA;           // Code for the Color Filter Array of the sensor
    #                            // CFA_NONE=0,(gray) CFA_VRI=1(gbrg/rggb),
    #                            // CFA_VRIV6=2(bggr/grbg), CFA_BAYER=3(gb/rg)
    #                            // CFA_BAYERFLIP=4 (rg/gb)
    #                       // high byte carries info about color/gray heads at
    #                            // v6 and v6.2
    #                           // Masks: 0x80000000: TLgray 0x40000000: TRgray
    #                            // 0x20000000: BLgray 0x10000000: BRgray
    #                            // Final adjustments after image processing:
    #  int32_t Bright;           // ---UPDF replaced by fOffset
    #  uint32_t Res21;           // ---TBI
    #  uint32_t AutoExpLevel;    // Level for autoexposure control
    #  uint32_t AutoExpSpeed;    // Speed for autoexposure control
    #  RECT AutoExpRect;         // Rectangle for autoexposure control
    #  WBGAIN WBGain[4];   // Gain adjust on R,B components, for white balance,
    #                            // at Recording
    #                            // 1.0 = do nothing,
    #                            // index 0: all image for v4,5,7...
    #                            // and TL head for v6, v6.2 (multihead)
    #                            // index 1, 2, 3 :
    #                            // TR, BL, BR for multihead
    #  int32_t Rotate;           // Rotate the image 0=do nothing
    #                            // +90=counterclockwise -90=clockwise
    #  WBGAIN WBView;     // White balance to apply on color interpolated Cines
    #  uint32_t RealBPP;         // Real number of bits per pixel for this cine
    #                            // 8 on 8 bit cameras
    #                            // (v3, 4, 5, 6, 42, 43, 51, 62, 72, 9)
    #                            // Phantom v7: 8 or 12
    #                            // 14 bit cameras 8, 10, 12, 14
    #                          // Pixels will be stored on 8 or 16 bit in files
    #                            // and in PC memory
    #                           //(if RealBPP>8 the storage will be on 16 bits)
    #  //First degree function to convert the 16 bits pixels to 8 bit
    #  //(for display or file convert)
    #  uint32_t Conv8Min;        // ---TBI
    #                            // Minimum value when converting to 8 bits
    #  uint32_t Conv8Max;        // ---UPDF replaced by fGain16_8
    #                            // Max value when converting to 8 bits
    #  int32_t FilterCode;       // ImageProcessing: area processing code
    #  int32_t FilterParam;      // ImageProcessing: optional parameter
    #  IMFILTER UF;        // User filter: a 3x3 or 5x5 user convolution filter
    #  uint32_t BlackCalSVer;    // Software Version used for Black Reference
    #  uint32_t WhiteCalSVer;    // Software Version used for White Calibration
    #  uint32_t GrayCalSVer;     // Software Version used for Gray Calibration
    #  bool32_t bStampTime;      // Stamp time (in continuous recording)
    #                            // 1 = absolute time, 3 = from trigger
    #                        // End of SETUP in software version 605 (Nov 2003)
    #  uint32_t SoundDest;  // Sound device 0: none, 1: Speaker, 2: sound board
    #  //Frame rate profile
    #  uint32_t FRPSteps;        // Suplimentary steps in frame rate profile
    #                            // 0 means no frame rate profile
    #
    #
    #
    #
    #  int32_t FRPImgNr[16];    // Image number where to change the rate and/or
    #                       // exposure allocated for 16 points (4 available in
    #                            // in v7)
    #  uint32_t FRPRate[16];     // New value for frame rate (fps)
    #  uint32_t FRPExp[16]       // New value for exposure
    #                            // (nanoseconds, not implemented in cameras)
    #  //Multicine partition
    #  int32_t MCCnt;            // Partition count (= cine count - 1)
    #                            // Preview cine does not need a partition
    #  float MCPercent[64];      // Percent of memory used for partitions
    #                            // Allocated for 64 partitions, 15 used in the
    #                            // current cameras
    #                        // End of SETUP in software version 606 (May 2004)
    #  // CALIBration on Current Image (CSR, current session reference)
    #  uint32_t CICalib;         // This cine or this stg is the result of
    #                            // a current image calibration
    #                           // masks: 1 BlackRef, 2 WhiteCalib, 4 GrayCheck
    #                            // Last cicalib done at the acqui params:
    #  uint32_t CalibWidth;      // Image dimensions
    #  uint32_t CalibHeight;
    #  uint32_t CalibRate;       // Frame rate (frames per second)
    #  uint32_t CalibExp;        // Exposure duration (nanoseconds)
    #  uint32_t CalibEDR;        // EDR (nanoseconds)
    #  uint32_t CalibTemp;       // Sensor Temperature
    #  uint32_t HeadSerial[4];   // Head serials for ethernet multihead cameras
    #                        // (v6.2) When multiple heads are saved in a file,
    #                           // the serials for existing heads are not zero
    #                         // When one head is saved in a file its serial is
    #                            // in HeadSerial[0] and the other head serials
    #                            // are 0xFFffFFff
    #                        // End of SETUP in software version 607 (Oct 2004)
    #  uint32_t RangeCode;  // Range data code: describes the range data format
    #  uint32_t RangeSize;       // Range data, per image size
    #  uint32_t Decimation;     // Factor to reduce the frame rate when sending
    #                            //the images to i3 external memory by fiber
    #                         //End of SETUP in software version 614 (Feb 2005)
    #  uint32_t MasterSerial;// Master camera Serial for external sync. 0 means
    #                            // none (this camera is not a slave of another
    #                            // camera)
    #                        // End of SETUP in software version 624 (Jun 2005)
    #  uint32_t Sensor;          // Camera sensor code
    #                        // End of SETUP in software version 625 (Jul 2005)
    #  //Acquisition parameters in nanoseconds
    #  uint32_t ShutterNs;       // Exposure, in nanoseconds
    #  uint32_t EDRShutterNs;    // EDRExp, in nanoseconds
    #  uint32_t FrameDelayNs;    // FrameDelay, in nanoseconds
    #                        // End of SETUP in software version 631 (Oct 2005)
    #  //Stamp outside the acquired image
    # //this increases the image size by adding a border with text information
    #  uint32_t ImPosXAcq;       // Acquired image horizontal offset in
    #                            // sideStamped image
    #  uint32_t ImPosYAcq;     // Acquired image vertical offset in sideStamped
    #                            // image
    #  uint32_t ImWidthAcq;      // Acquired image width (different value from
    #                            // ImWidth if sideStamped file)
    #  uint32_t ImHeightAcq;     // Acquired image height (different value from
    #                            // ImHeight if sideStamped file)
    #  char Description[MAXLENDESCRIPTION];//User description or comments
    #                            //(enlarged to 4096 characters)
    #                        // End of SETUP in software version 637 (Jul 2006)
    #  bool32_t RisingEdge;      // TRUE rising, FALSE falling
    #  uint32_t FilterTime;      // time constant
    #  bool32_t LongReady;       // If TRUE the Ready is 1 from the start
    #                            // to the end of recording (needed for signal
    #                            // acquisition)
    #  bool32_t ShutterOff;  // Shutter off - to force maximum exposure for PIV
    #                        // End of SETUP in software version 658 (Mar 2008)
    #  uint8_t Res4[16];         // ---TBI
    #                        // End of SETUP in software version 663 (May 2008)
    #  bool32_t bMetaWB;         // pixels value does not have WB applied
    #                            // (or any other processing)
    #  int32_t Hue;              //---UPDF replaced by float fHue
    #                            // hue corection (degrees: -180 ...180)
    #                        // End of SETUP in software version 671 (May 2009)
    #  int32_t BlackLevel;       // Black level in the raw pixels
    #  int32_t WhiteLevel;       // White level in the raw pixels
    #  char LensDescription[256];// text with the producer, model,
    #                            // focal range etc ...
    #  float LensAperture;       // aperture f number
    #  float LensFocusDistance;  // distance where the objects are in focus in
    #                        // meters, not available from Canon motorized lens
    #  float LensFocalLength;    // current focal length; (zoom factor)
    #                        // End of SETUP in software version 691 (Jul 2010)
    #  float fOffset;            // [-1.0, 1.0], neutral 0.0;
    #                            // 1.0 means shift by the maximum pixel value
    #  float fGain;              // [0.0, Max], neutral 1.0;
    #  float fSaturation;        // [0.0, Max], neutral 1.0;
    #  float fHue;               // [-180.0, 180.0] neutral 0;
    #                      // degrees and fractions of degree to rotate the hue
    #  float fGamma;             // [0.0, Max], neutral 1.0; global gamma
    #                            // (or green gamma)
    #  float fGammaR;          / per component gammma (to be added to the field
    #                            // Gamma)
    #                            // 0 means neutral
    #  float fGammaB;
    #  float fFlare;             // [-1.0, 1.0], neutral 0.0;
    #                            // 1.0 means shift by the maximum pixel value
    #                            // pre White Balance offset
    #  float fPedestalR;         // [-1.0, 1.0], neutral 0.0;
    #                            // 1.0 means shift by the maximum pixel value
    #                            // after gamma offset
    #  float fPedestalG;
    #  float fPedestalB;
    #  float fChroma;            // [0.0, Max], neutral 1.0;
    #                            // chrominance adjustment (after gamma)
    #  char ToneLabel[256];
    #  int32_t TonePoints;
    #  float fTone[32*2];        // up to 32 points + 0.0,0.0 1.0,1.0
    #                            // defining a LUT using spline curves
    #  char UserMatrixLabel[256];
    #  bool32_t EnableMatrices;
    #  float cmUser[9];          // RGB color matrix
    #  bool32_t EnableCrop;  // The Output image will contains only a rectangle
    #                            // portion of the input image
    #  RECT CropRect;
    #  bool32_t EnableResample;// Resample image to a desired output Resolution
    #  uint32_t ResampleWidth;
    #  uint32_t ResampleHeight;
    #  float fGain16_8;        // Gain coefficient used when converting to 8bps
    #                            // Input pixels (bitdepth>8) are multiplied by
    #                           // the factor: fGain16_8 * (2**8 / 2**bitdepth)
    #                        // End of SETUP in software version 693 (Oct 2010)
    #
    #  uint32_t FRPShape[16];    // 0: flat, 1 ramp
    #  TC TrigTC;                // Trigger frame SMPTE time code and user bits
    #  float fPbRate;         // Video playback rate (fps) active when the cine
    #                           // was captured
    #  float fTcRate;          // Playback rate (fps) used for generating SMPTE
    #                           // time code
    #                        // End of SETUP in software version 701 (Apr 2011)
    #
    #  char CineName[256];       // Cine name
    #                        // End of SETUP in software version 702 (May 2011)
    #  float fGainR;             // Per component gain - user adjustment
    #  float fGainG;             // [0.0, Max], neutral 1.0;
    #  float fGainB;
    #  float cmCalib[9]//RGB color calibration matrix bringing camera pixels to
    #                  // rec 709. It includes the white balance set into the
    #                  // ph16 cameras using fWBTemp and fWBCc and the original
    #                  // factory calibration. The cine player should decompose
    #                  //this matrix in two components: a diagonal one with the
    #                  //white balance to be applied before interpolation and a
    #                  // normalized one to be applied after interpolation.
    #
    #  float fWBTemp;  // White balance based on color temperature and color
    #  float fWBCc;    // temperature and color compensation index.
    #                  // Its effect is included in the cmCalib.
    #  char CalibrationInfo[1024];
    #                  // Original calibration matrices: used to calculate
    #                  // cmCalib in the camera
    #  char OpticalFilter[1024];
    #                  // Optical filter matrix: used to calculate cmCalib in
    #                  // the camera
    #                  // End of SETUP in software version 709 (September 2011)
    #  MAXSTDSTRSZ = 256
    #  char GpsInfo[MAXSTDSTRSZ];
    #                  //Current position and status info as received from a
    #                  // GPS receiver
    #  char Uuid[MAXSTDSTRSZ];// Unique cine identifier
    #                      // End of SETUP in software version 719 (March 2012)
    #  char CreatedBy[MAXSTDSTRSZ];
    #                     // The name of the application that created this cine
    #                     // End of SETUP in software version 720 (March 2012)
    #  uint32_t RecBPP;   // Acquisition bit depth. It can be 8, 10, 12 or 14
    #  uint16_t LowestFormatBPP;
    #                   // Description of the minimum format out of all formats
    #  uint16_t LowestFormatQ//the images of this cine have been represented on
    #                         //since the moment of acquisition.
    #                    //End of SETUP in software version 731 (February 2013)
    #  float fToe;               // Controls the gamma curve in the blacks.
    #                            // Neutral is 1.0f.
    #                            // Decreasing fToe lifts the blacks, while
    #                            // increasing it compresses them
    #
    #  uint32_t LogMode;         // Configures the log mode.
    #                            // 0 - log mode disabled.
    #                            // 1, 2, etc - log mode enabled.
    #                        //If log mode enabled, gain, gamma, the pedestals,
    #                            // r/g/b gains, offset and flare are inactive.
    #
    #  char CameraModel[MAXSTDSTRSZ]; // Camera model string.
    #                            // End of SETUP in software version 742
    #
    #  uint32_t WBType;      //For raw color cines, it describes how meta WB is
    #                            // stored.
    #                         // If bit 0 is set - wb is stored in WBGain field
    #                        //If bit 1 is set - wb is stored in fWBTemp, fWBCc
    #                            // fields.
    #  float fDecimation;  //Decimation coefficient employed when this cine was
    #                            // saved to file.
    #                            // End of SETUP in software version 745
    #  uint32_t MagSerial;    // The serial of the magazine where the cine was
    #                            // recorded or stored (if any, null otherwise)
    #  uint32_t CSSerial; //Cine Save serial: The serial of the device (camera,
    #                            // cine station) used to save the cine to file
    #  double dFrameRate;      //High precision acquisition frame rate, replace
    #                            // uint32_t FrameRate
    #                            // End of SETUP in software version 751
    #  uint32_t SensorMode    //Camera sensor Mode used to define what mode the
    #                            // Sensor is in
    #                      //This is currently valid for the Flex4K, VEO 4K and
    #                            // VIRGO V2640
    #                      //it is undefined for all other camera (should be 0)
    #                        // End of SETUP in software version 771 (Dec 2017)
    return cin_settings


def read_image_header(filename: str, bit_pos: int, verbose: bool = False):
    """
    Read the image header of a .cin file

    Josse Rueda: jrrueda@us.es

    @param filename: name of the .cin file (full path to the file)
    @param bit_pos: position of the file where the image header starts
    @param verbose: flag to display content of the header
    @return cin_image_header: Image header (dictionary), see in-line comments
    for a full description of each dictionary field
    """
    # --- Open file and go to the position of the image header
    fid = open(filename, 'r')
    if verbose:
        print('Reading .cin image header')
    fid.seek(bit_pos)
    # read the image header
    # number of bytes of the structure
    cin_image_header = {'biSize': np.fromfile(fid, 'uint32', 1),
                        'biWidth': np.fromfile(fid, 'int32', 1),
                        'biHeight': np.fromfile(fid, 'int32', 1),
                        'biPlanes': np.fromfile(fid, 'uint16', 1)}
    # Size of the frame in pixels
    # number of planes for the target devices
    if cin_image_header['biPlanes'] != 1:
        print('The number of planes not equal 1, possible problem reading the '
              'file')
    # number of bits per pixels
    cin_image_header['biBitCount'] = np.fromfile(fid, 'uint16', 1)
    # Compression of the image
    cin_image_header['biCompression'] = np.fromfile(fid, 'uint32', 1)
    # Size of the image in bytes
    cin_image_header['biSizeImage'] = np.fromfile(fid, 'uint32', 1)
    # Pixels per meter in the sensor
    cin_image_header['biXPelsPerMeter'] = np.fromfile(fid, 'int32', 1)
    cin_image_header['biYPelsPerMeter'] = np.fromfile(fid, 'int32', 1)
    # Color settings
    cin_image_header['biClrUsed'] = np.fromfile(fid, 'uint32', 1)
    cin_image_header['biClrImportant'] = np.fromfile(fid, 'uint32', 1)
    if cin_image_header['biClrImportant'] == 4096 and verbose:
        print('The actual depth used for the scale is 2^12')
    fid.close()
    # return and print
    # Print if needed
    if verbose:
        for y in cin_image_header:
            print(y, ':', cin_image_header[y])
    return cin_image_header


def read_time_base(filename: str, header: dict, settings: dict):
    """
    Read the time base of the .cin video (relative to the trigger)

    Jose Rueda: jrrueda@us.es

    @param filename: name of the file to open (full path)
    @param header: header created by the function read_header
    @param settings: setting dictionary created by read_settings
    @return cin_time: np array with the time point for each recorded frame
    """
    # Open file and go to the position of the image header
    fid = open(filename, 'r')
    fid.seek(header['OffSetup'] + settings['Length'])

    # There are different blocks order in a different way depending on the
    # version of the camera, we are interested in the block identified with a
    # 1002 number
    id_number = 0
    cumulate_size: int = 0
    while id_number != 1002:
        size_of_the_block = np.fromfile(fid, 'uint32', 1)
        cumulate_size = cumulate_size + size_of_the_block

        id_number = np.fromfile(fid, 'uint16', 1)

        if id_number == 1002:
            np.fromfile(fid, 'uint16', 1)  # Reserved number
            dummy = np.fromfile(fid, 'uint32',
                                int(2 * header['ImageCount'][:]))
            cin_time: float = np.float64(dummy[1::2]) - \
                np.float64(header['TriggerTime']['seconds']) + \
                (np.float64(dummy[0::2]) -
                 np.float64(header['TriggerTime']['fractions'])) / 2.0 ** 32
            # return
            fid.close()
            return cin_time
        fid.seek(header['OffSetup'] + settings['Length'] + cumulate_size)


def read_frame_cin(cin_object, frames_number, limitation: bool = True,
                   limit: int = 2048):
    """
    Read frames from a .cin file

    Jose Rueda Rueda: jrrueda@us.es

    @param cin_object: Video Object with the file information
    @param frames_number: np array with the frame numbers to load
    @param limitation: maximum size allowed to the output variable,
    in Mbytes, to avoid overloading the memory trying to load the whole
    video of 100 Gb
    @param limit: bool flag to decide if we apply the limitation of we
    operate in mode: YOLO
    @return M: 3D numpy array with the frames M[px,py,nframes]
    """
    # --- Section 0: Initial checks
    flags_negative = frames_number < 0
    flags_above = frames_number > cin_object.header['ImageCount']
    if (np.sum(flags_above) + np.sum(flags_negative)) > 0:
        print('The requested frames are not in the file!!!')
        return 0
    # Check the output size:
    nframe = np.size(frames_number)
    if nframe > 1:
        # weight of the output array in megabytes
        talla = cin_object.imageheader['biWidth'] * \
                cin_object.imageheader['biHeight'] * nframe * 2 / 1024 / 1024
        # If the weight is too much, stop (to do not load tens og Gb of video
        # in the memory and kill the computer
        if (talla > limit) and limitation:
            print('Output will be larger than the limit, stopping')
            return 0

    # --- Section 1: Get frames position
    # Open file and go to the position of the image header
    fid = open(cin_object.file, 'r')
    fid.seek(cin_object.header['OffImageOffsets'])

    if cin_object.header['Version'] == 0:  # old format
        position_array = np.fromfile(fid, 'int32',
                                     int(cin_object.header['ImageCount']))
    else:
        position_array = np.fromfile(fid, 'int64',
                                     int(cin_object.header['ImageCount']))
    # -------------------------------------------------------------------------
    # ---  Section 2: Read the images

    # Image size from header information
    # First look for the actual number of pixel used to store information of
    # the counts in each pixel. If this is 8 or less the images will be saved
    # in 8 bit format, else, 16 bits will be used
    size_info = cin_object.settings['RealBPP']
    if size_info <= 8:
        BPP = 8  # Bits per pixel
        data_type = 'uint8'
    else:
        BPP = 16  # Bits per pixel
        data_type = 'uint16'

    # Pre-allocate output array
    # To be in line with old FILDGUI and be able to use old FILD calibration
    # the matrix will be [height,width]
    M = np.zeros((int(cin_object.imageheader['biHeight']),
                  int(cin_object.imageheader['biWidth']), nframe),
                 dtype=data_type)
    img_size_header = cin_object.imageheader['biWidth'] * \
        cin_object.imageheader['biHeight'] * BPP / 8  # In bytes
    npixels = cin_object.imageheader['biWidth'] * \
        cin_object.imageheader['biHeight']
    # Read the frames
    for i in range(nframe):
        #  Go to the position of the file
        iframe = frames_number[i]  # - cin_object.header['FirstImageNo']
        fid.seek(position_array[iframe])
        #  Skip header of the frame
        length_annotation = np.fromfile(fid, 'uint32', 1)
        fid.seek(position_array[iframe] + length_annotation - 4)
        #  Read frame
        image_size = np.fromfile(fid, 'uint32', 1)  # In bytes
        if image_size != img_size_header:
            print(image_size)
            print(img_size_header)
            print('Image sizes (in bytes) does not coincides')
            return 0

        M[:, :, i] = np.reshape(np.fromfile(fid, data_type,
                                            int(npixels)),
                                (int(cin_object.imageheader['biWidth']),
                                 int(cin_object.imageheader['biHeight'])),
                                order='F').transpose()
    fid.close()
    return M.squeeze()  # eliminate extra dimension in case we have just loaded
    #                     one frame


# -----------------------------------------------------------------------------
# --- Methods for the .png files
# -----------------------------------------------------------------------------
def read_data_png(path):
    """
    Read info for a case where the measurements are stored as png

    Jose Rueda Rueda: jrrueda@us.es

    Return a series of dictionaries similar to the case of a cin file,
    with all the info we can extract from the png

    @param path: path to the folder where the pngs are located
    @return time_base: time base of the frames (s)
    @return image_header: dictionary containing the info about the image size,
    and shape
    @return header: Header 'similar' to the case of a cin file
    @return settings: dictionary similar to the case of the cin file (it only
    contain the exposition time)
    """
    # Look for a png to extract the file and a .txt for the time information
    f = []
    look_for_png = True
    for file in os.listdir(path):
        if file.endswith('.txt'):
            f.append(os.path.join(path, file))
        if file.endswith('.png') and look_for_png:
            dummy = io.imread(os.path.join(path, file))
            si = dummy.shape
            imageheader = {'biWidth': si[0],
                           'biHeight': si[1]}
            look_for_png = False
    # If no png was found, raise and error
    if look_for_png:
        print('No .png files in the folder...')
        return 0, 0, 0, 0
    n_files = len(f)
    if n_files == 0:
        print('no txt file with the information found in the directory!!')
        return 0, 0, 0, 0
    elif n_files > 1:
        print('Several txt files found...')
        return 0, 0, 0, 0
    else:
        dummy = np.loadtxt(f[0], skiprows=2, comments='(')
        header = {'ImageCount': int(dummy[-1, 0])}
        settings = {'ShutterNs': dummy[0, 2] / 1000.0}
        time_base = dummy[:, 3]
        # Sometimes the camera break and set the says that all the frames
        # where recorded at t = 0... in this case just assume a time base on
        # the basis of the exposition time
        std_time = np.std(time_base)
        # Test if all the exposue time was the same
        if std_time < 1e-2 and np.mean(time_base) < 0.1:
            time_base = np.linspace(0, dummy[-1, 0] * dummy[0, 2] / 1000,
                                    int(dummy[-1, 0]))
            print('Caution!! the experimental time base was broken, a time '
                  'base has been generated on the basis of theexposure time')
        elif np.mean(time_base) < 0.1:
            raise Exception('The time base was broken!!!')
    # extract the shot number from the path
    header['shot'] = int(path[-5:])

    return header, imageheader, settings, time_base[:].flatten()


def rgb2gray(rgb):
    """Transform rgb images to gray"""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def load_png_files(filename: str):
    """
    Load the png with an order compatible with IDL

    IDL load things internally in a way different from python. In order the new
    suite to be compatible with all FILD calibrations of the last 15 years,
    an inversion should be done to load png in the same way as IDL

    @param filename: full path pointing to the png
    """

    dummy = io.imread(filename)
    if len(dummy.shape) > 2:     # We have an rgb png, transform it to gray
        dummy = rgb2gray(dummy)

    return dummy[::-1, :]


def read_frame_png(video_object, frames_number=None, limitation: bool = True,
                   limit: int = 2048):
    """
    Read .png files

    Jose Rueda: jrrueda@us.es

    @param video_object: Video class with the info of the video
    @param frames_number: array with the number of the frames to be loaded,
    if none, all frames will be loaded
    @param limitation: if we want to set a limitation of the size we can load
    @param limit: Limit to the size, in megabytes
    @return M: array of frames, [px in x, px in y, number of frames]
    """
    # Frames would have a name as shot-framenumber.png example: 30585-001.png
    print('Reading PNG files')
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
                      video_object.header['ImageCount']), dtype=np.float32)
        counter = 0
        for file in os.listdir(video_object.path):
            if file.endswith('.png'):
                M[:, :, counter] = load_png_files(
                    os.path.join(video_object.path, file)).astype(np.float32)
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
                      len(frames_number)), dtype=np.float32)

        for file in os.listdir(video_object.path):
            if file.endswith('.png'):
                current_frame = current_frame + 1
                if current_frame in frames_number:
                    pngname = os.path.join(video_object.path, file)
                    dummy = load_png_files(pngname)
                    M[:, :, counter] = dummy.astype(np.float32)
                    counter = counter + 1
        print('Number of loaded frames: ', counter)
    return M


# -----------------------------------------------------------------------------
# --- mp4 files
# -----------------------------------------------------------------------------
def read_mp4_file(file, verbose: bool = True):
    """
    Read frames and time base from an mp4 file

    Jose Rueda: jrrueda@us.es

    @param file: full path to the file
    """
    # --- Open the video file
    vid = cv2.VideoCapture(file)

    # --- Get the number of frames in the video
    nf = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # --- Get the frame rate
    fr = vid.get(cv2.CAP_PROP_FPS)
    if verbose:
        print('We will load: ', nf, ' frames')
        print('Frame rate: ', fr)
    # --- Read the frames
    nx = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ny = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    time = np.zeros(nf)
    frames = np.zeros((nx, ny, nf))
    counter = 0
    success = True
    while success:
        success, image = vid.read()
        if success:
            time[counter] = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # if frame is in rgb, transform to gray
            if len(image.shape) > 1:
                image = rgb2gray(image)
            frames[:, :, counter] = image
            counter += 1
    return {'nf': nf, 'nx': nx, 'ny': ny, 'frames': frames, 'tframes': time}


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class Video:
    """
    Class with the information of the recorded video

    The header, image header, settings and time base will be stored here,
    the frames itself not, we can not work with 100Gb of data in memory!!!
    """

    def __init__(self, file: str = None, diag: str = 'FILD', shot=None,
                 diag_ID: int = 1):
        """
        Initialise the class

        @param file: For the initialization, file (full path) to be loaded),
        if the path point to a .cin file, the .cin file will be loaded. If
        the path points to a folder, the program will look for png files or
        tiff files inside (tiff coming soon). If none, a window will be open to
        select a file
        @param shot: Shot number, if is not given, the program will look for it
        in the name of the loaded file
        """
        # If no file was given, open a graphical user interface to select it.
        if file is None:
            root = tk.Tk()    # To close the window after the selection
            root.withdraw()
            filename = tk.filedialog.askopenfilename()
            if filename == '':
                raise Exception('You must select a file!!!')
            # If we select a png or tif, we need the folder, not the file
            if filename.endswith('.png') or filename.endswith('.tif'):
                file, name_png = os.path.split(filename)
            else:
                file = filename
        # Initialise some variables
        ## Type of video
        self.type_of_file = None
        ## Loaded experimental data
        self.exp_dat = {'frames': None,   # Loaded frames
                        'tframes': None,  # Timebase of the loaded frames
                        'nframes': None}  # Frame numbers of the loaded frames
        ## Remapped data
        self.remap_dat = None
        ## Time traces: space reservation for the future
        self.time_trace = None
        ## Diagnostic used to record the data
        self.diag = diag
        ## Diagnostic ID (FILD number)
        self.diag_ID = diag_ID
        ## Shot number
        self.shot = shot
        if shot is None:
            self.guess_shot(file, ssdat.shot_number_length)

        # Fill the object depending if we have a .cin file or not
        if os.path.isfile(file):
            ## Path to the file and filename
            self.path, self.file_name = os.path.split(file)
            ## Name of the file (full path)
            self.file = file

            # Check if the file is actually a .cin file
            if file[-4:] == '.cin' or file[-5:] == '.cine':
                ## Header dictionary with the file info
                self.header = read_header(file)
                ## Settings dictionary
                self.settings = read_settings(file, self.header['OffSetup'])
                ## Image Header dictionary
                self.imageheader = read_image_header(file, self.header[
                    'OffImageHeader'])
                ## Time array
                self.timebase = read_time_base(file, self.header,
                                               self.settings)
                self.type_of_file = '.cin'
            elif file[-4:] == '.png' or file[-4:] == '.tif':
                raise Exception('To load png or tif, just give the folder')
            elif file[-4:] == '.mp4':
                dummy = read_mp4_file(file, verbose=False)
                # mp4 files are handle different as they are suppose to be just
                # a dummy temporal format, not the one to save our real exp
                # data, so the video will be loaded all from here and not
                # reading specific frame will be used
                self.timebase = dummy['tframes']
                self.exp_dat['frames'] = dummy['frames']
                self.exp_dat['tframes'] = dummy['tframes']
                self.exp_dat['nframes'] = np.arange(dummy['nf'])
                self.type_of_file = '.mp4'
            else:
                raise Exception('Not recognised file extension')
        elif os.path.isdir(file):
            ## path to the file
            self.path = file
            # Do a quick run for the folder looking of .tiff or .png files
            f = []
            for (dirpath, dirnames, filenames) in os.walk(self.path):
                f.extend(filenames)
                break

            for i in range(len(f)):
                if f[i][-4:] == '.png':
                    self.type_of_file = '.png'
                    print('Found PNG files!')
                    break
                elif f[i][-4:] == '.tif':
                    self.type_of_file = '.tif'
            # if we do not have .png or tiff, give an error
            if self.type_of_file != '.png' and self.type_of_file != '.tiff':
                print('No .pgn or .tiff files found')
                return

            # If we have a .png file, a .txt must be present with the
            # information of the exposure time and from a basic frame we can
            # load the file size
            if self.type_of_file == '.png':
                self.header, self.imageheader, self.settings,\
                    self.timebase = read_data_png(self.path)
        if self.type_of_file is None:
            raise Exception('Not file found!')

    def guess_shot(self, file, shot_number_length):
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
            self.shot = int(list[flags])
        elif ntrues == 2:
            # Maybe just the file is saved in a folder named as the shot, so we
            # can have a second positive here
            options = list[flags]
            if options[0] == options[1]:
                self.shot = int(options[0])
        elif ntrues == 0:
            er = 'No shot number found in the name of the file\n'
            er2 = 'Give the shot number as input when loading the file'
            raise Exception(er + er2)
        else:
            er = 'Several possibles shot number were found\n'
            er2 = 'Give the shot number as input when loading the file'
            print('Possible shot numbers ', list[flags])
            raise Exception(er + er2)

    def read_frame(self, frames_number=None, limitation: bool = True,
                   limit: int = 2048, internal: bool = True, t1: float = None,
                   t2: float = None):
        """
        Call the read_frame function

        Just a wrapper to call the read_frame function, depending of the
        format in which the experimental data has been recorded

        @param frames_number: np array with the frame numbers to load
        @param limitation: maximum size allowed to the output variable,
        in Mbytes, to avoid overloading the memory trying to load the whole
        video of 100 Gb
        @param limit: bool flag to decide if we apply the limitation or if we
        operate in YOLO mode
        @param internal: If True, the frames will be stored in the 'frames'
        variable of the video object. Else, it will be returned just as output
        (usefull if you need to load another frame and you do not want to
        overwrite your frames already loaded)
        @param t1: Initial time to load frames (alternative to frames number)
        @param t2: Final time to load frames (alternative to frames number), if
        just t1 is given , only one frame will be loaded
        @return M: 3D numpy array with the frames M[px,py,nframes]
        """
        # --- Select frames to load
        if (frames_number is not None) and (t1 is not None):
            raise Exception('You cannot give frames number and time')
        elif (t1 is not None) and (t2 is None):
            frames_number = np.array([np.argmin(abs(self.timebase-t1))])
        elif (t1 is not None) and (t2 is not None):
            it1 = np.argmin(abs(self.timebase-t1))
            it2 = np.argmin(abs(self.timebase-t2))
            frames_number = np.arange(start=it1, stop=it2+1, step=1)
        # else:
        #     raise Exception('Something went wrong, check inputs')

        if self.type_of_file == '.cin':
            if internal:
                self.exp_dat['frames'] = \
                    read_frame_cin(self, frames_number, limitation=limitation,
                                   limit=limit)
                self.exp_dat['tframes'] = self.timebase[frames_number]
                self.exp_dat['nframes'] = frames_number
            else:
                M = read_frame_cin(self, frames_number, limitation=limitation,
                                   limit=limit)
                return M
        elif self.type_of_file == '.png':
            if internal:
                self.exp_dat['frames'] = \
                    read_frame_png(self, frames_number, limitation=limitation,
                                   limit=limit)
                self.exp_dat['tframes'] = \
                    self.timebase[frames_number].flatten()
                self.exp_dat['nframes'] = frames_number
            else:
                M = read_frame_png(self, frames_number, limitation=limitation,
                                   limit=limit)
                return M
        else:
            raise Exception('Not initialised file type?')
        return

    def subtract_noise(self, t1: float = None, t2: float = None, frame=None):
        """
        Subtract noise from camera frames

        Jose Rueda: jrrueda@us.es

        This function subtract the noise from the experimental camera frames.
        Two main ways exist: if t1 and t2 are provided, the noise will be
        be considered as the average in this range. If 'frame' is given,
        the noise to be subtracted will be considered to be directly 'frame'

        A new variable: 'original frames' will be created, where the original
        frames will be loaded, in case one wants to revert the noise
        subtraction

        @param t1: Minimum time to average the noise
        @param t2: Maximum time to average the noise
        @param frame: Optional, frame containing the noise to be subtracted
        """
        print('.--. ... ..-. -')
        print('Substracting noise')
        # Get shape and data type of the experimental data
        nx, ny, nt = self.exp_dat['frames'].shape
        original_dtype = self.exp_dat['frames'].dtype
        # Calculate the noise frame, if needed:
        if (t1 is not None) and (t2 is not None):
            it1 = np.argmin(abs(self.exp_dat['tframes'] - t1))
            it2 = np.argmin(abs(self.exp_dat['tframes'] - t2))
            print('Using frames from the video')
            print(str(it2 - it1 + 1), ' frames will be used to average noise')
            frame = np.mean(self.exp_dat['frames'][:, :, it1:(it2 + 1)],
                            dtype=original_dtype, axis=2)
        else:
            print('Using noise frame provider by the user')
            nxf, nyf = frame.shape
            if (nxf != nx) or (nyf != ny):
                raise Exception('The noise frame has not the correct shape')
        # Create the original frame array:
        if 'original_frames' not in self.exp_dat:
            self.exp_dat['original_frames'] = self.exp_dat['frames'].copy()
        else:
            print('original frames already present, not making new copy')
        # Subtract the noise
        for i in range(nt):
            # Perform the subtraction in float to avoid negative values in uint
            dummy = \
                self.exp_dat['frames'][:, :, i].squeeze().astype(np.float64) -\
                frame.astype(np.float64)
            # Set negative counts to zero:
            dummy[dummy < 0] = 0.
            self.exp_dat['frames'][:, :, i] = dummy.astype(original_dtype)
        return

    def return_to_original_frames(self):
        """
        Place in self.exp_dat['frames'] the real experimental frames

        Jose Rueda: jrrueda@us.es

        Useful if some operation was performed and we want to place
        again the original frames at self.exp_dat['frames']
        """
        if 'original_frames' not in self.exp_dat:
            raise Exception('A copy of the original frames was not found!')
        else:
            self.exp_dat['frames'] = self.exp_dat['original_frames'].copy()
        return

    def remap_loaded_frames(self, calibration, shot, options: dict = {}):
        """
        Remap all loaded frames in the video object

        @param    calibration: Calibration object (see LibMap)
        @type:    type

        @param    shot: Shot number
        @type:    type

        @param    options: Options for the remapping routine. See
        remap_all_loaded_frames_XXXX in the LibMap package for a full
        description
        @type:    dict

        @return:  write in the object a dictionary containing with:
            -# options: Options used for the remapping
            -# frames: Remaped frames
            -# time: time associated to the remapped points
            -# xaxis: xaxis of the remapped frames
            -# xlabel: name of the xaxis of he remaped frame (pitch for FILD)
            -# yaxis: xaxis of the remapped frames
            -# ylabel: name of the yaxis of he remaped frame (r for FILD)
            -# sprofx: signal integrated over the y range given by options
            -# sprofy: signal integrated over the x range given by options
        """

        if self.diag == 'FILD':
            self.remap_dat, opt = \
                ssmap.remap_all_loaded_frames_FILD(self, calibration, shot,
                                                   **options)
            self.remap_dat['options'] = opt

    def plot_frame(self, frame_number=None, ax=None, fig=None, ccmap=None,
                   strike_map=None, t: float = None):
        """
        Plot a frame from the loaded frames

        Not recommended for general use (it can be slow and it is not very
        customizable) it is though just for a quick plot

        Notice: If ax is given, fig should be also given

        @param frame_number: Number of the frame to plot, relative to the video
        file, optional
        @param fig: Figure where the frame must be drawn
        @param ax: Axes where to plot, is none, just a new axes will be created
        @param ccmap: colormap to be used, if none, Gamma_II from IDL
        @param strike_map: StrikeMap to be plotted on top (optional)
        @return ax: the axes where the frame has been drawn
        @return fig: the figure where the frame has been drawn
        """
        # --- Check inputs:
        if (frame_number is not None) and (t is not None):
            raise Exception('Do not give frame number and time!')
        if (frame_number is None) and (t is None):
            raise Exception("Didn't you want to plot something?")
        # --- Load the frames
        # If we use the frame number explicitly
        if frame_number is not None:
            if len(self.exp_dat['nframes']) == 1:
                if self.exp_dat['nframes'] == frame_number:
                    dummy = self.exp_dat['frames'].squeeze()
                    tf = float(self.exp_dat['tframes'])
                else:
                    raise Exception('Frame not loaded')
            else:
                frame_index = self.exp_dat['nframes'] == frame_number
                if np.sum(frame_index) == 0:
                    raise Exception('Frame not loaded')
                dummy = self.exp_dat['frames'][:, :, frame_index].squeeze()
                tf = float(self.exp_dat['tframes'][frame_index])
        # If we give the time:
        if t is not None:
            it = np.argmin(abs(self.exp_dat['tframes'] - t))
            tf = self.exp_dat['tframes'][it]
            dummy = self.exp_dat['frames'][:, :, it].squeeze()
        # --- Check the colormap
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap
        # --- Check the axes to plot
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(dummy, origin='lower', cmap=cmap)
        ax.set_title('t = ' + str(round(tf, 4)) + (' s'))
        # --- Plot the StrikeMap
        if strike_map is not None:
            strike_map.plot_pix(ax=ax)

        return fig, ax

    def plot_frames_slider(self):
        "Plot the frames with a slider which allows to select the desired time"
        root = tk.Tk()
        ApplicationShowVid(root, self.exp_dat)
        root.mainloop()
        root.destroy()

    def plot_remaped_slider(self, ccmap=None):
        """
        Creates a plot with some sliders to see the remapped data

        Jose Rueda Rueda: jrrueda@us.es

        @param ccmap: colormap to be used, if none, Gamma_II will be used
        """
        # @todo solve this in a more elegant way
        global exp_dat
        global remap_dat
        global ax
        global cmap
        global cbar_frame
        global cbar_remap
        global gyr
        global pitch
        # global rfild
        # global zfild
        global pa
        global phi
        global theta
        global pEner
        global plt_param
        global options_plt_pitch
        global options_plt_remap
        global options_plt_gyr
        fig, ax = plt.subplots(2, 2)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap
        # Get the extreme of the counts, for the scale
        max_rep = np.max(self.remap_dat['frames'])
        min_rep = np.min(self.remap_dat['frames'])
        max_fra = np.max(self.exp_dat['frames'])
        min_fra = np.min(self.exp_dat['frames'])
        # --- Initialise the four plots
        # - Camera
        plot_frame = ax[0, 0].imshow(self.exp_dat['frames'][:, :, 0],
                                     origin='lower', cmap=cmap)
        # plt_param = {'marker': 'None'}
        cbar_frame = fig.colorbar(plot_frame, ax=ax[0, 0])
        # - Remaped info
        # @todo: think in giving as output this transposed
        dummy = self.remap_dat['frames'][:, :, 0].squeeze().T
        plot_remap = ax[0, 1].contourf(self.remap_dat['xaxis'],
                                       self.remap_dat['yaxis'],
                                       dummy, levels=20, cmap=cmap)
        options_plt_remap = {'xlabel': self.remap_dat['xlabel'],
                             'ylabel': self.remap_dat['ylabel']}
        dum = ssplt.axis_beauty(ax[0, 1], options_plt_remap)
        cbar_remap = fig.colorbar(plot_remap, ax=ax[0, 1])
        # - pitch profile
        plot_pitch = ax[1, 0].plot(self.remap_dat['xaxis'],
                                   self.remap_dat['sprofx'][:, 0])
        options_plt_pitch = {'ylabel': 'Counts',
                             'xlabel': self.remap_dat['xlabel'] + ' [' +
                             self.remap_dat['xunits'] + ']',
                             'grid': 'both'}
        dum = ssplt.axis_beauty(ax[1, 0], options_plt_pitch)
        # - gir profile
        plot_gyr = ax[1, 1].plot(self.remap_dat['yaxis'],
                                 self.remap_dat['sprofy'][:, 1])
        options_plt_gyr = {'ylabel': 'Counts',
                           'xlabel': self.remap_dat['ylabel'] + ' [' +
                           self.remap_dat['yunits'] + ']',
                           'grid': 'both'}
        dum = ssplt.axis_beauty(ax[1, 1], options_plt_gyr)
        # --- Set the sliders
        # - Time slider
        axcolor = 'k'
        dt = self.exp_dat['tframes'][1] - self.exp_dat['tframes'][0]
        axtime = plt.axes([0.15, 0.97, 0.75, 0.02], facecolor=axcolor)
        stime = Slider(axtime, 'Time [s]: ', self.exp_dat['tframes'][0],
                       self.exp_dat['tframes'][-1], valfmt='%.3f',
                       valinit=self.exp_dat['tframes'][0], valstep=dt)
        # - Colormap for the frame
        axmax_frame = plt.axes([0.15, 0.94, 0.30, 0.02], facecolor=axcolor)
        axmin_frame = plt.axes([0.15, 0.91, 0.30, 0.02], facecolor=axcolor)
        smin_frame = Slider(axmax_frame, 'Min [#]: ', 0.75 * min_fra,
                            1.25 * max_fra, valinit=min_fra)
        smax_frame = Slider(axmin_frame, 'Max [#]: ',  0.75 * min_fra,
                            1.25 * max_fra, valinit=max_fra)
        # - Colormap for the remap
        axmax_rep = plt.axes([0.60, 0.94, 0.30, 0.02], facecolor=axcolor)
        axmin_rep = plt.axes([0.60, 0.91, 0.30, 0.02], facecolor=axcolor)
        smin_rep = Slider(axmax_rep, 'Min [#]: ', 0.75 * min_rep,
                          1.25 * max_rep, valinit=min_rep)
        smax_rep = Slider(axmin_rep, 'Max [#]: ',  0.75 * min_rep,
                          1.25 * max_rep, valinit=max_rep)

        exp_dat = self.exp_dat
        remap_dat = self.remap_dat

        # function to update the plots:
        def update(val):
            """Update the sliders"""

            ####
            # Get the time
            t = stime.val
            it = np.argmin(abs(exp_dat['tframes']-t))
            # Get the limit of the colorbars
            frame_min = smin_frame.val
            frame_max = smax_frame.val
            rep_min = smin_rep.val
            rep_max = smax_rep.val
            # - Plot the frame
            ax[0, 0].clear()
            dummy = exp_dat['frames'][:, :, it].squeeze()
            plot_frame = ax[0, 0].imshow(dummy, origin='lower', cmap=cmap,
                                         vmin=frame_min, vmax=frame_max)
            cbar_frame.update_normal(plot_frame)
            # Get and plot the strike map
            # name = ssFILDSIM.find_strike_map(rfild, zfild, phi[it],
            #                                  theta[it], pa.StrikeMaps,
            #                                  pa.FILDSIM)
            # map = ssmap.StrikeMap(0, os.path.join(pa.StrikeMaps, name))
            # map.calculate_pixel_coordinates(cal)
            # map.plot_pix(ax[0, 0], plt_param=plt_param)

            # - Plot the remap
            ax[0, 1].clear()
            dummy = remap_dat['frames'][:, :, it].squeeze()
            plot_remap = ax[0, 1].contourf(remap_dat['xaxis'],
                                           remap_dat['yaxis'],
                                           dummy.T, levels=20,
                                           cmap=cmap, vmin=rep_min,
                                           vmax=rep_max)
            dum = ssplt.axis_beauty(ax[0, 1], options_plt_remap)
            cbar_remap.update_normal(plot_remap)
            # - Update the profiles
            # Plot the profiles
            ax[1, 0].clear()
            ax[1, 0].plot(remap_dat['xaxis'],
                          remap_dat['sprofx'][:, it])
            dum = ssplt.axis_beauty(ax[1, 0], options_plt_pitch)
            ax[1, 1].clear()
            ax[1, 1].plot(remap_dat['yaxis'],
                          remap_dat['sprofy'][:, it])
            dum = ssplt.axis_beauty(ax[1, 1], options_plt_gyr)

            fig.canvas.draw_idle()
        stime.on_changed(update)
        smin_frame.on_changed(update)
        smax_frame.on_changed(update)
        smin_rep.on_changed(update)
        smax_rep.on_changed(update)
        plt.show()
        return fig, ax

    def plot_profiles_in_time(self, ccmap=None, plt_params: dict = {}, t=None,
                              nlev: int = 50, cbar_tick_format: str = '%.1E',
                              normalise=False):
        """
        Creates a plot with the evolution of the profiles

        Jose Rueda Rueda: jrrueda@us.es

        @param ccmap: colormap to be used, if none, Gamma_II will be used
        @param plt_params: params for the function axis beauty plt
        @param t: time, if present, just a line plot for the profiles for that
        time will be used
        @param nlev: Number of levels for the contourf plots
        @param cbar_tick_format: format for the colorbar ticks
        @todo: substitute pprofmin and max, also with rl or we will have a
        future bug
        """
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap

        if 'fontsize' not in plt_params:
            plt_params['fontsize'] = 16

        if t is None:  # 2d plots
            # Gyroradius profiles
            fig1, ax1 = plt.subplots()
            cont = ax1.contourf(self.remap_dat['tframes'],
                                self.remap_dat['yaxis'],
                                self.remap_dat['sprofy'], nlev, cmap=cmap)
            cbar = plt.colorbar(cont, format=cbar_tick_format)
            cbar.set_label('Counts [a.u.]', fontsize=plt_params['fontsize'])
            cbar.ax.tick_params(labelsize=plt_params['fontsize'] * .8)
            # Write the shot number and detector id
            gyr_level = self.remap_dat['yaxis'][-1] -\
                0.1*(self.remap_dat['yaxis'][-1]
                     - self.remap_dat['yaxis'][0])  # Jut a nice position
            tpos1 = self.remap_dat['tframes'][0] + 0.05 * \
                (self.remap_dat['tframes'][-1] - self.remap_dat['tframes'][0])
            tpos2 = self.remap_dat['tframes'][0] + 0.95 * \
                (self.remap_dat['tframes'][-1] - self.remap_dat['tframes'][0])
            FS = plt_params['fontsize']
            if self.diag == 'FILD':
                plt.text(tpos1, gyr_level, '#' + str(self.shot),
                         horizontalalignment='left', fontsize=0.9*FS,
                         color='w', verticalalignment='bottom')
                plt.text(tpos1, gyr_level,
                         str(self.remap_dat['options']['pprofmin']) + 'º to '
                         + str(self.remap_dat['options']['pprofmax']) + 'º',
                         horizontalalignment='left', fontsize=0.9*FS,
                         color='w', verticalalignment='top')
                plt.text(tpos2, gyr_level, self.diag + str(self.diag_ID),
                         horizontalalignment='right', fontsize=0.9*FS,
                         color='w', verticalalignment='bottom')
                plt_params['xlabel'] = 'Time [s]'
                plt_params['ylabel'] = self.remap_dat['ylabel'] + ' [' +\
                    self.remap_dat['yunits'] + ']'
            ax1 = ssplt.axis_beauty(ax1, plt_params)
            plt.tight_layout()
            # Pitch profiles in time
            fig2, ax2 = plt.subplots()
            cont = ax2.contourf(self.remap_dat['tframes'],
                                self.remap_dat['xaxis'],
                                self.remap_dat['sprofx'], nlev, cmap=cmap)
            cbar = plt.colorbar(cont, format=cbar_tick_format)
            cbar.set_label('Counts [a.u.]', fontsize=plt_params['fontsize'])
            cbar.ax.tick_params(labelsize=plt_params['fontsize'] * .8)
            # Write the shot number and detector id
            level = self.remap_dat['xaxis'][-1] -\
                0.1*(self.remap_dat['xaxis'][-1]
                     - self.remap_dat['xaxis'][0])  # Jut a nice position
            if self.diag == 'FILD':  # Add a labal with the integration range
                plt.text(tpos1, level, '#' + str(self.shot),
                         horizontalalignment='left', fontsize=0.9*FS,
                         color='w', verticalalignment='bottom')
                plt.text(tpos1, level,
                         str(self.remap_dat['options']['rprofmin']) + 'cm to '
                         + str(self.remap_dat['options']['rprofmax']) + 'cm',
                         horizontalalignment='left', fontsize=0.9*FS,
                         color='w', verticalalignment='top')
                plt.text(tpos2, level, self.diag + str(self.diag_ID),
                         horizontalalignment='right', fontsize=0.9*FS,
                         color='w', verticalalignment='bottom')
                plt_params['xlabel'] = 'Time [s]'
                plt_params['ylabel'] = self.remap_dat['xlabel'] + ' [' +\
                    self.remap_dat['xunits'] + ']'
            ax2 = ssplt.axis_beauty(ax2, plt_params)
            plt.tight_layout()
        else:  # The line plots:
            # Set the grid option for plotting
            if 'grid' not in plt_params:
                plt_params['grid'] = 'both'
            # see if the input time is an array:
            try:
                t.size
            except AttributeError:
                try:
                    len(t)
                    t = np.array(t)
                except TypeError:
                    t = np.array([t])
            # Open the figure
            fig, (ax1, ax2) = plt.subplots(1, 2)
            for tf in t:
                # find the frame we want to plot
                it = np.argmin(abs(self.remap_dat['tframes'] - tf))
                # Plot the gyroradius profile:
                if normalise:
                    y = self.remap_dat['sprofy'][:, it]
                    y /= y.max()
                else:
                    y = self.remap_dat['sprofy'][:, it]
                ax1.plot(self.remap_dat['yaxis'], y,
                         label='t = {0:.3f}s'.format(
                            self.remap_dat['tframes'][it]))
                # Plot the pitch profile
                if normalise:
                    y = self.remap_dat['sprofx'][:, it]
                    y /= y.max()
                else:
                    y = self.remap_dat['sprofx'][:, it]
                ax2.plot(self.remap_dat['xaxis'], y,
                         label='t = {0:.3f}s'.format(
                            self.remap_dat['tframes'][it]))
            if self.diag == 'FILD':
                title = '#' + str(self.shot) + ' ' +\
                    str(self.remap_dat['options']['pprofmin']) + 'º to ' +\
                    str(self.remap_dat['options']['pprofmax']) + 'º'
                plt_params['xlabel'] = self.remap_dat['ylabel'] + ' [' +\
                    self.remap_dat['yunits'] + ']'
                plt_params['ylabel'] = 'Counts [a.u.]'
                ax1.set_title(title, fontsize=plt_params['fontsize'])
            ax1 = ssplt.axis_beauty(ax1, plt_params)
            ax1.legend()
            if self.diag == 'FILD':
                title = '#' + str(self.shot) + ' ' +\
                    str(self.remap_dat['options']['rprofmin']) + 'cm to ' +\
                    str(self.remap_dat['options']['rprofmax']) + 'cm'
                plt_params['xlabel'] = self.remap_dat['xlabel'] + ' [' +\
                    self.remap_dat['xunits'] + ']'
                plt_params['ylabel'] = 'Counts [a.u.]'
                ax2.set_title(title, fontsize=plt_params['fontsize'])
            ax2.legend()
            ax2 = ssplt.axis_beauty(ax2, plt_params)
            plt.tight_layout()
        return

    def export_remap(self):
        """
        Export the dictionary containing the remapped data

        Jose Rueda Rueda: jrrueda@us.es
        """

        # Test if the folder
        name = os.path.join(pa.ScintSuite, 'Results', str(self.shot) + '_'
                            + self.diag + '_remap.nc')
        print('Saving results in: ', name)
        # Write the data:
        with netcdf.netcdf_file(name, 'w') as f:
            f.history = 'Done with version ' + version

            # Save shot number
            f.createDimension('number', 1)
            shot = f.createVariable('shot', 'i', ('number', ))
            shot[:] = self.shot
            shot.units = ' '
            shot.long_name = 'Shot number'

            # Save the time of the remapped frames
            f.createDimension('tframes', len(self.remap_dat['tframes']))
            time = f.createVariable('tframes', 'float64', ('tframes', ))
            time[:] = self.remap_dat['tframes']
            time.units = 's'
            time.long_name = 'Time'

            # Save the pitches
            f.createDimension('xaxis', len(self.remap_dat['xaxis']))
            xaxis = f.createVariable('xaxis', 'float64', ('xaxis', ))
            xaxis[:] = self.remap_dat['xaxis']
            xaxis.units = self.remap_dat['xunits']
            xaxis.long_name = self.remap_dat['xlabel']

            # Save the gyroradius
            f.createDimension('yaxis', len(self.remap_dat['yaxis']))
            yaxis = f.createVariable('yaxis', 'float64', ('yaxis', ))
            yaxis[:] = self.remap_dat['yaxis']
            yaxis.units = self.remap_dat['yunits']
            yaxis.long_name = self.remap_dat['ylabel']

            # Save the remapped data
            frames = f.createVariable('frames', 'float64',
                                      ('xaxis', 'yaxis', 'tframes'))
            frames[:, :, :] = self.remap_dat['frames']
            frames.units = 'Counts per axis area'
            frames.long_name = 'Remapped frames'

            # Save the modulus of the magnetic field at the FILD positon
            bfield = f.createVariable('bfield', 'float64', ('tframes', ))
            bfield[:] = self.remap_dat['bfield']
            bfield.units = 'T'
            bfield.long_name = 'Field at detector'

            # Save the temporal evolution of the profiles
            sprofx = f.createVariable('sprofx', 'float64',
                                      ('xaxis', 'tframes'))
            sprofx[:, :] = self.remap_dat['sprofx']
            sprofx.units = 'a.u.'
            sprofx.long_name = self.remap_dat['sprofxlabel']

            sprofy = f.createVariable('sprofy', 'float64',
                                      ('yaxis', 'tframes'))
            sprofy[:, :] = self.remap_dat['sprofy']
            sprofy.units = 'a.u.'
            sprofy.long_name = self.remap_dat['sprofylabel']

            # Save the specific FILD variables
            if self.diag == 'FILD':
                # Detector orientation
                theta = f.createVariable('theta', 'float64', ('tframes', ))
                theta[:] = self.remap_dat['theta']
                theta.units = '{}^o'
                theta.long_name = 'theta'

                phi = f.createVariable('phi', 'float64', ('tframes', ))
                phi[:] = self.remap_dat['phi']
                phi.units = '{}^o'
                phi.long_name = 'phi'

                # Options used for the remapping
                rmin = f.createVariable('rmin', 'float64', ('number', ))
                rmin[:] = self.remap_dat['options']['rmin']
                rmin.units = 'cm'
                rmin.long_name = 'Minimum r_l for the remap'

                rmax = f.createVariable('rmax', 'float64', ('number', ))
                rmax[:] = self.remap_dat['options']['rmax']
                rmax.units = 'cm'
                rmax.long_name = 'Maximum r_l for the remap'

                dr = f.createVariable('dr', 'float64', ('number', ))
                dr[:] = self.remap_dat['options']['dr']
                dr.units = 'cm'
                dr.long_name = 'dr_l for the remap'

                dp = f.createVariable('dp', 'float64', ('number', ))
                dp[:] = self.remap_dat['options']['dp']
                dp.units = '{}^o'
                dp.long_name = 'dp for the remap'

                pmin = f.createVariable('pmin', 'float64', ('number', ))
                pmin[:] = self.remap_dat['options']['pmin']
                pmin.units = '{}^o'
                pmin.long_name = 'Minimum pitch for the remap'

                pmax = f.createVariable('pmax', 'float64', ('number', ))
                pmax[:] = self.remap_dat['options']['pmax']
                pmax.units = '{}^o'
                pmax.long_name = 'Maximum pitch for the remap'

                pprofmin = f.createVariable('pprofmin', 'float64', ('number',))
                pprofmin[:] = self.remap_dat['options']['pprofmin']
                pprofmin.units = '{}^o'
                pprofmin.long_name = 'Minimum pitch to integrate the remap'

                pprofmax = f.createVariable('pprofmax', 'float64', ('number',))
                pprofmax[:] = self.remap_dat['options']['pprofmax']
                pprofmax.units = '{}^o'
                pprofmax.long_name = 'Maximum pitch to integrate the remap'

                rprofmin = f.createVariable('rprofmin', 'float64', ('number',))
                rprofmin[:] = self.remap_dat['options']['rprofmin']
                rprofmin.units = 'cm'
                rprofmin.long_name = 'Minimum r_l to integrate the remap'

                rprofmax = f.createVariable('rprofmax', 'float64', ('number',))
                rprofmax[:] = self.remap_dat['options']['rprofmax']
                rprofmax.units = 'cm'
                rprofmax.long_name = 'Maximum r_l to integrate the remap'

                rfild = f.createVariable('rfild', 'float64', ('number',))
                rfild[:] = self.remap_dat['options']['rfild']
                rfild.units = 'm'
                rfild.long_name = 'R FILD position'

                zfild = f.createVariable('zfild', 'float64', ('number',))
                zfild[:] = self.remap_dat['options']['zfild']
                zfild.units = 'm'
                zfild.long_name = 'z FILD position'

                alpha = f.createVariable('alpha', 'float64', ('number',))
                alpha[:] = self.remap_dat['options']['alpha']
                alpha.units = '{}^o'
                alpha.long_name = 'alpha orientation'

                beta = f.createVariable('beta', 'float64', ('number',))
                beta[:] = self.remap_dat['options']['beta']
                beta.units = '{}^o'
                beta.long_name = 'beta orientation'


class ApplicationShowVid:
    """Class to show the frames"""

    def __init__(self, master, data):
        """
        Create the window with the sliders

        @param master: Tk() opened
        @param data: the dictionary of experimental frames or remapped ones
        """
        # Save here the data
        self.data = data
        # Create a container
        frame = tk.Frame(master)
        # Create the time slider
        t = data['tframes']
        print(t.shape)
        print(t.size)
        dt = t[1] - t[0]
        self.tSlider = tk.Scale(master, from_=t[0], to=t[-1], resolution=dt,
                                command=self.plot_frame)
        self.tSlider.pack(side='right')
        # create the quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.pack(side='bottom')

        fig = Figure()
        ax = fig.add_subplot(111)
        self.image = ax.imshow(data['frames'][:, :, 0].squeeze(),
                               origin='lower')
        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

    def plot_frame(self, t):
        """Get a plot the frame"""
        t0 = np.float64(t)
        it = np.argmin(abs(self.data['tframes'] - t0))
        dummy = self.data['frames'][:, :, it].squeeze()
        self.image.set_data(dummy)
        self.canvas.draw()
