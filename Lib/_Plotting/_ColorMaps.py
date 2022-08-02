"""
Contains custom color maps included in the suite

Jose Rueda Rueda: jrrueda@us.es

Contains:
    -Gamma_II: Similar to IDL colormap with the same name
    -Cai: Color map with the colors of Cadiz
"""
from matplotlib.colors import LinearSegmentedColormap
__all__ = ['Gamma_II', 'Cai']


def Gamma_II(n=256):
    """
    Gamma II colormap

    This function creates the colormap that coincides with the
    Gamma_II_colormap of IDL.

    Jose Rueda: jrrueda@us.es

    @param n: numbers of levels of the output colormap
    """
    cmap = LinearSegmentedColormap.from_list(
        'mycmap', ['black', 'blue', 'red', 'yellow', 'white'], N=n)
    return cmap


def Cai(n=256):
    """
    Cai II colormap

    This is a kind of an easter egg

    Jose Rueda: jrrueda@us.es

    @param n: numbers of levels of the output colormap
    """
    cmap = LinearSegmentedColormap.from_list(
        'mycmap', ['blue', 'yellow'], N=n)
    return cmap
