"""
Library to manipulate UFILES.

Taken from Giovanni Tardini's repository (git@ipp.mpg.de)

Adapted to read generic UFILEs by Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import datetime
import shutil
import os

def wr_for(arr_in, fmt='%13.6E', n_lin=6):
    """
    Prepare and array to the appropriate format to be written into a UFILE.

    Giovanni Tardini - git@ipp.mpg.de

    @param arr_in: array to be written in an UFILE.
    @param fmt: format of the output.
    @param n_lin: number of maximum numbers per line.
    """

    arr_flat = arr_in.T.ravel()
    nx = len(arr_flat)
    out_str=''
    for jx in range(nx):
        out_str += (fmt %arr_flat[jx])
        if (jx%n_lin == n_lin - 1):
            out_str += '\n'
    # Line break after writing data, but no double break
    if (nx%n_lin != 0):
        out_str += '\n'

    return out_str


def ssplit(ll):
    """
    Parse an input line and separates them and put the resulting numbers into
    a list.

    Giovanni Tardini - git@ipp.mpg.de

    @param ll: one line of text containing numbers.
    """

    tmp = ll.replace('-', ' -')
    tmp = tmp.replace('e -', 'e-')
    tmp = tmp.replace('E -', 'E-')
    slist = tmp.split()
    a = [float(i) for i in slist]

    return a


def lines2fltarr(lines, dtyp=None):
    """
    Converts a collection of lines into a flattened array.

    Giovanni Tardini - git@ipp.mpg.de

    @param lines: set of lines to be converted into a 1D array.
    @param dtyp: type of the output.
    """
    data = []
    for line in lines:
        data += ssplit(line)
    if dtyp is None:
        dtyp = np.float32

    return np.array(data, dtype=dtyp)


def fltarr_len(lines, nx: int, dtyp=None):
    """
    Converts a collection of lines into a flattened array, providing a maximum
    length for the data.

    Giovanni Tardini - git@ipp.mpg.de

    @param lines: set of lines to be converted into a 1D array.
    @param nx: maximum lenght of the output array.
    @param dtyp: type of the output.
    """
    data = []
    for jlin, line in enumerate(lines):
        data += ssplit(line)
        if len(data) >= nx:
            break
    if dtyp is None:
        dtyp = np.float32

    return jlin+1, np.array(data, dtype=dtyp)


class ufile:
    """
    Class to handle the UFILE content: read/write and parsing the dimensions.
    """
    def __init__(self, fin: str=None):
        """
        Initializes the class for the UFILE. If a name is provided, the object
        proceeds to read its content.

        Giovanni Tardini - git@ipp.mpg.de

        @param fin: filename of the file to read. If not provided, the class
        is initialized as empty.
        """
        self.f = {}
        if fin is not None:
            self.read(fin)


    def read(self, fin):
        """
        Reads the content of the UFILE into the class.

        Giovanni Tarding - git@ipp.mpg.de

        @param fin: input filename to read
        """

        # Loading all the lines from the file.
        with open(fin, 'r') as f:
            lines = f.readlines()

        # Preparing the dimensions.
        dims = []
        coords = ['X', 'Y', 'Z']
        self.comment = ''

        n_coords = 0
        jend = np.nan

        # Loop over the lines.
        for jline, lin in enumerate(lines):
            if '-SHOT #' in lin or '; Shot #' in lin:
                a = lin.split()
                self.shot = int(a[0][:5])
                ndim = int(a[1])

            if '-INDEPENDENT VARIABLE LABEL' in lin or '; Independent variable' in lin:
                for coord in coords:
                    svar = coord + '-'
                    if svar in lin:
                        self.__dict__[coord] = {}
                        co_d = self.__dict__[coord]
                        co_d['label'] = lin.split(';-')[0][1: ]
                        co_d['name'] = co_d['label'][:20].strip()
                        co_d['unit'] = co_d['label'][20:].strip()
                        n_coords += 1
                if n_coords == 0:
                    if '-INDEPENDENT' in lin:
                        self.X['label'] = lin.split(';-')[0][1: ]
                    else:
                        self.X['label'] = lin.split('; Independent')[0][1: ]
                    self.X['name'] = self.X['label'][:20].strip()
                    self.X['unit'] = self.X['label'][20:].strip()
                    n_coords = 1

            for var in coords:
                svar1 = '-# OF %s PTS-' %var
                svar2 = ';-# of radial pts  %s' %var
                if svar1 in lin or svar2 in lin:
                   a = lin.split()
                   dims.append(int(a[0]))

            if '-DEPENDENT VARIABLE LABEL-' in lin:
                self.f['label'] = lin.split(';-')[0][1: ]
                self.f['name'] = self.f['label'][:20].strip()
                self.f['unit'] = self.f['label'][20:].strip()
            try:
                flt_lin = ssplit(lines[jline])
            except:
                if 'flt_lin' not in locals():
                    jstart = jline+1
                else:
                    if np.isnan(jend):
                        jend = jline

            if jline > jend:
                self.comment += lin
        self.comment = self.comment[1:]

        # ... Grid data

        if ndim != n_coords:
            raise Exception('Inconsistency in the number of independent'+\
                            'variables dim=%d nvar=%d', ndim, n_coords)
            return

        data = lines2fltarr(lines[jstart: jend])

        ind_split = np.cumsum(dims)
        if ndim == 1:
            self.X['data'], farr = np.split(data, ind_split)
        elif ndim == 2:
            self.X['data'], self.Y['data'], farr = np.split(data, ind_split)
        elif ndim == 3:
            self.X['data'], self.Y['data'], self.Z['data'], farr = \
                np.split(data, ind_split)
        self.f['data'] = farr.reshape(dims[::-1]).T


    def average(self, axis: int=0):
        """
        Perform an average of the data along the input axis.

        Giovanni Tarding - git@ipp.mpg.de

        @param axis: axis along which to perform the average.
        """
        self.ext = self.ext + '_AVG%d' %axis
        if axis == 0:
            self.X['data'] = np.atleast_1d(np.nanmean(self.X['data']))
        elif axis == 1:
            self.Y['data'] = np.atleast_1d(np.nanmean(self.Y['data']))
        elif axis == 2:
            self.Z['data'] = np.atleast_1d(np.nanmean(self.Z['data']))
        tmp = np.nanmean(self.f['data'], axis=axis)
        shape = list(self.f['data'].shape)
        shape[axis] = 1
        self.f['data'] = tmp.reshape(shape)


    def write(self, fout: str, dev: str='AUGD'):
        """
        Writes the content of the class into a file.

        Giovanni Tarding - git@ipp.mpg.de

        @param udir: directory to write the UFILE.
        @param dev:  target directory to write the UFILE accordingly.
        """
        self.shot = int(self.shot)

        if hasattr(self, 'comment'):
            comment = self.comment
        else:
            comment = ''

        if os.path.isfile(fout):
            ufbak = fout
            jext = 1
            while os.path.isfile(ufbak):
                ufbak = '%s.%d' %(fout, jext)
                jext += 1
            shutil.move(fout, ufbak)

        # Dimensions check
        dims = []
        coords = []
        if hasattr(self, 'X'):
            coords.append('X')
            dims.append(len(self.X['data']))
        if hasattr(self, 'Y'):
            coords.append('Y')
            dims.append(len(self.Y['data']))
        if hasattr(self, 'Z'):
            coords.append('Z')
            dims.append(len(self.Z['data']))

        ndim = len(dims)
        if ndim != self.f['data'].ndim:
            raise Exception('Data ndim inconsistent with n of independent variables')

        for jdim, dim in enumerate(dims):
            if dim != self.f['data'].shape[jdim]:
                raise ValueError('The %dth index has inconsistent'%jdim + \
                                 ' grid vs data dimension: %d, %d'%(dim,
                                   self.f['data'].shape[jdim] ))
                return

        # Header
        now = datetime.datetime.now()
        ufs = '  %5d%4s %1i 0 6              ;-SHOT #- F(X) DATA -UF%1iDWR- %s\n'\
             %(self.shot,  dev,  ndim,  ndim, now.strftime('%d%b%Y'))
        ufs += ''.ljust(30) + ';-SHOT DATE-  UFILES ASCII FILE SYSTEM\n'

        if hasattr(self, 'scalar'):
            nscal = len(self.scalar)
            ufs += (' %2d' %nscal).ljust(30) + \
                ';-NUMBER OF ASSOCIATED SCALAR QUANTITIES-\n'
            for jscal in range(nscal):
                lbl = self.scalar[jscal]['label']
                val = self.scalar[jscal]['data']
                ufs += ('%11.4E' %val).ljust(30) + ';-SCALAR,  LABEL FOLLOWS:\n'
                ufs += lbl.ljust(30) + '\n'
        else:
            ufs += '  0'.ljust(30) +';-NUMBER OF ASSOCIATED SCALAR QUANTITIES-\n'
        if ndim > 0:
            if (ndim == 1):
                if 'unit' in self.X.keys() and 'name' in self.X.keys():
                    ufs += '%-20s%-10s;-INDEPENDENT VARIABLE LABEL-\n' \
                        %(self.X['name'], self.X['unit'])
                elif 'label' in self.X.keys():
                    ufs += '%-30s;-INDEPENDENT VARIABLE LABEL-\n' %self.X['label']
                else:
                    raise Exception('X must have either "name"+"unit" or "label" key')
            else:
                for coord in coords:
                    co_d = self.__dict__[coord]
                    if 'unit' in co_d.keys() and 'name' in co_d.keys():
                        ufs += '%-20s%-10s;-INDEPENDENT VARIABLE LABEL: ' \
                            %(co_d['name'], co_d['unit'])
                    elif 'label' in co_d.keys():
                        ufs += '%-30s;-INDEPENDENT VARIABLE LABEL: ' %co_d['label']
                    else:
                        raise Exception('%s must have either "name"+"unit" or "label" key'%coord)
                    ufs += '%1s-\n' %coord

        if 'unit' in self.f.keys():
            ufs += '%-20s%-10s;-DEPENDENT VARIABLE LABEL-\n' \
            %(self.f['name'], self.f['unit'])
        else:
            ufs += '%-30s;-DEPENDENT VARIABLE LABEL-\n'   %self.f['label']
        ufs += '0'.ljust(30) + ';-PROC CODE- 0:RAW 1:AVG 2:SM. 3:AVG+SM\n'

        # Grids

        for jcoord, coord in enumerate(coords):
            ufs += ('   %7d' %dims[jcoord]).ljust(30) + ';-# OF %1s PTS-\n' %coord
        for coord in coords:
            data = self.__dict__[coord]['data']
            ufs += wr_for(data)

        # Data

        ufs += wr_for(self.f['data'])

        ufs += ';----END-OF-DATA-----------------COMMENTS:-----------\n'
        ufs = ufs.replace('\n', '\n ')
        ufs += comment + '\n'

        with open(fout, 'w') as fid:
            fid.write(ufs)