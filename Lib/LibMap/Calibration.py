"""Calibration and database objects."""
import numpy as np


class CalibrationDatabase:
    """Database of parameter to align the scintillator."""

    def __init__(self, filename: str, n_header: int = 5):
        """
        Read the calibration database, to align the strike maps.

        See database page for a full documentation of each field

        @author Jose Rueda Rueda: jrrueda@us.es

        @param filename: Complete path to the file with the calibrations
        @param n_header: Number of header lines
        @return database: Dictionary containing the database information
        """
        ## Name of file with the data
        self.file = filename
        ## Header of the file
        self.header = []
        ## Dictionary with the data from the calibration. See @ref database
        ## for a full description of the meaning of each field
        self.data = {'ID': [], 'camera': [], 'shot1': [], 'shot2': [],
                     'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
                     'deg': [], 'cal_type': [], 'diag_ID': []}
        # Open the file
        with open(filename) as f:
            for i in range(n_header):
                # Lines with description
                self.header = self.header + [f.readline()]
            # Database itself
            for line in f:
                dummy = line.split()
                self.data['ID'] = self.data['ID'] + [int(dummy[0])]
                self.data['camera'] = self.data['camera'] + [dummy[1]]
                self.data['shot1'] = self.data['shot1'] + [int(dummy[2])]
                self.data['shot2'] = self.data['shot2'] + [int(dummy[3])]
                self.data['xshift'] = self.data['xshift'] + [float(dummy[4])]
                self.data['yshift'] = self.data['yshift'] + [float(dummy[5])]
                self.data['xscale'] = self.data['xscale'] + [float(dummy[6])]
                self.data['yscale'] = self.data['yscale'] + [float(dummy[7])]
                self.data['deg'] = self.data['deg'] + [float(dummy[8])]
                self.data['cal_type'] = self.data['cal_type'] + [dummy[9]]
                self.data['diag_ID'] = self.data['diag_ID'] + [int(dummy[10])]

    def write_database_to_txt(self, file: str = None):
        """
        Write database into a txt.

        If no name is given, the name of the loaded file will be used but a
        'new' will be added. Example: if the file from where the info has
        been loaded is 'calibration.txt' the new file would be
        'calibration_new.txt'. This is just to be save and avoid overwriting
        the original database.

        @param file: name of the file where to write the results
        """
        if file is None:
            file = self.file[:-4] + '_new.txt'
        with open(file, 'w') as f:
            # Write the header
            for i in range(len(self.header)):
                f.write(self.header[i])
            # Write the database information
            for i in range(len(self.data['ID'])):
                line = str(self.data['ID'][i]) + ' ' + \
                       self.data['camera'][i] + ' ' + \
                       str(self.data['shot1'][i]) + ' ' + \
                       str(self.data['shot2'][i]) + ' ' + \
                       str(self.data['xshift'][i]) + ' ' + \
                       str(self.data['yshift'][i]) + ' ' + \
                       str(self.data['xscale'][i]) + ' ' + \
                       str(self.data['yscale'][i]) + ' ' + \
                       str(self.data['deg'][i]) + ' ' + \
                       self.data['cal_type'][i] + ' ' + \
                       str(self.data['diag_ID'][i]) + ' ' + '\n'
                f.write(line)
            print('File: ' + file + ' writen')

    def get_calibration(self, shot, camera, cal_type, diag_ID):
        """
        Give the calibration parameter of a precise database entry.

        @param shot: Shot number for which we want the calibration
        @param camera: Camera used
        @param cal_type: Type of calibration we want
        @param diag_ID: ID of the diagnostic we want
        @return cal: CalParams() object
        """
        flags = np.zeros(len(self.data['ID']))
        for i in range(len(self.data['ID'])):
            if (self.data['shot1'][i] <= shot) * \
                    (self.data['shot2'][i] >= shot) * \
                    (self.data['camera'][i] == camera) * \
                    (self.data['cal_type'][i] == cal_type) * \
                    (self.data['diag_ID'][i] == diag_ID):
                flags[i] = True

        n_true = sum(flags)

        if n_true == 0:
            raise Exception('No entry find in the database, revise database')
        elif n_true > 1:
            print('Several entries fulfill the condition')
            print('Possible entries:')
            print(self.data['ID'][flags])
            raise Exception()
        else:
            dummy = np.argmax(np.array(flags))
            cal = CalParams()
            cal.xscale = self.data['xscale'][dummy]
            cal.yscale = self.data['yscale'][dummy]
            cal.xshift = self.data['xshift'][dummy]
            cal.yshift = self.data['yshift'][dummy]
            cal.deg = self.data['deg'][dummy]

        return cal


class CalParams:
    """
    Information to relate points in the camera sensor the scintillator.

    In a future, it will contain the correction of the optical distortion and
    all the methods necessary to correct it.
    """

    def __init__(self):
        """Initialize the class"""
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        ## pixel/cm in the x direction
        self.xscale = 0.0
        ## pixel/cm in the y direction
        self.yscale = 0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.xshift = 0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.yshift = 0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg = 0.0
        ## Camera type
        self.camera = ''

    def print(self):
        """Print calibration"""
        print('xscale: ', self.xscale)
        print('yscale: ', self.yscale)
        print('xshift: ', self.xshift)
        print('yshift: ', self.yshift)
        print('deg: ', self.deg)
