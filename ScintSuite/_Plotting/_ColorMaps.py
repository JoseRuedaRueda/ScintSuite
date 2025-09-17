"""
Contains custom color maps included in the suite

Jose Rueda Rueda: jrrueda@us.es

Contains:
    -Gamma_II: Similar to IDL colormap with the same name
    -Gamma_III: Same as Gamma_II, but with grey at the bottom for contrast
    -Cai: Color map with the colors of Cadiz
"""
from matplotlib.colors import LinearSegmentedColormap
__all__ = ['Gamma_II', 'Gamma_III', 'Cai']

# Deprecated
# def Gamma_II(n=256):
#     """
#     Gamma II colormap

#     This function creates the colormap that coincides with the
#     Gamma_II_colormap of IDL.

#     Jose Rueda: jrrueda@us.es

#     :param  n: numbers of levels of the output colormap
#     """
#     cmap = LinearSegmentedColormap.from_list(
#         'mycmap', ['black', 'blue', 'red', 'yellow', 'white'], N=n)
#     return cmap

def Gamma_II(n=256):
    """
    Gamma_II colormap

    Creates the colormap very similar with the Gamma_II of IDL

    Alex Reyner: alereyvinn@alum.us.es

    :param  n: numbers of levels of the output colormap
    """
    cmap = LinearSegmentedColormap.from_list(
        'mycmap', ['black', 'blue', 'purple', 
                   'red', 'orange', 'yellow', 'white'], N=n)
    return cmap


def Gamma_III(n=256):
    """
    Gamma_II colormap with extra color in the lower range for higher contrast

    Alex Reyner: alereyvinn@alum.us.es

    :param  n: numbers of levels of the output colormap
    """
    color_positions = [0.0, 1/6/2, 1/6*1, 1/6*2, 1/6*3, 1/6*4, 1/6*5, 1]
    colors = ['black', 'silver', 'blue', 'purple', 
              'red', 'orange', 'yellow', 'white']

    cmap = LinearSegmentedColormap.from_list(
        "mycmap", list(zip(color_positions, colors)),N=n)
    return cmap



def Cai(n=256):
    """
    Cai II colormap

    This is a kind of an easter egg

    Jose Rueda: jrrueda@us.es

    :param  n: numbers of levels of the output colormap
    """
    cmap = LinearSegmentedColormap.from_list(
        'mycmap', ['blue', 'yellow'], N=n)
    return cmap
