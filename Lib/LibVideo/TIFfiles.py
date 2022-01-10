"""
Routines to read .tif files.

Written by Jose Rueda: jrrueda@us.es

These routines are just a wrapper for standard python methods just to leave the
data in the same order (colums and rows) from the old IDL FILD analysis
routines, in order to preserve the database variables

Under development
"""
from skimage import io                     # To load images
import Lib.LibVideo.AuxFunctions as aux


def read_data(path):
    """To Be implemented."""
    pass


def load_tiff(filename: str):
    """
    Load the tiff files

    @param filename: full path pointing to the tiff
    @return frame: loaded frame

    @ToDo: Check that the order is the correct one
    """
    dummy = io.imread(filename)
    if len(dummy.shape) > 2:     # We have an rgb tiff, transform it to gray
        dummy = aux.rgb2gray(dummy)

    return dummy[::-1, :]
