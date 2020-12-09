"""Contain physical constants and camera information

Basically here are hard-cored almost all the parameters of the suite
"""
# Physics constants
ec = 1.602e-19  # Electron charge, in C
mp = 938.272e6  # Mass of the proton, in eV/c^2
c = 3.0e8         # Speed of light in m/s

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from the paper:
# J. Ayllon-Guerola et al 2019 JINST14 C10032
# @todo find beta angles for the others FILD
fild1 = {'alpha': 0.0,   # Alpha angle [deg], see paper
         'beta': -12.0,  # beta angle [deg], see FILDSIM doc
         'sector': 8,    # The sector where FILD is located
         'r': 2.180,     # Radial position [m]
         'z': 0.3,       # Z position [m]
         'phi_tor': 169.75}  # Toroidal position, [deg]

fild2 = {'alpha': 0.0, 'beta': 1800.0, 'sector': 3, 'r': 2.180,
         'z': 0.3, 'phi_tor': 57.25}

fild3 = {'alpha': 72.0, 'beta': 1800.0, 'sector': 13, 'r': 1.975,
         'z': 0.765, 'phi_tor': 282.25}

fild4 = {'alpha': 0.0, 'beta': -12.0, 'sector': 8, 'r': 2.035,
         'z': -0.462, 'phi_tor': 169.75}

fild5 = {'alpha': -45.0, 'beta': 1800.0, 'sector': 7, 'r': 1.772,
         'z': -0.798, 'phi_tor': 147.25}

FILD = (fild1, fild2, fild3, fild4, fild5)


class Camera:
    """Class containing the properties of the cameras"""

    def __init__(self, model):
        """
        Initialise the class

        @param model: Model of the used camera
        """
        if model == 'VGA_Pixelfly':
            ## todo find the units in which pixel size ie given
            self.params = {'camera_name': 'VGA Pixelfly', 'nx_pixels': 640,
                           'ny_pixels': 480, 'pixel_xsize': 9.9e-4,
                           'pixel_ysize': 9.9e-4, 'quantum_efficiency': 0.40,
                           'f_analog_digital': 6.5, 'dynamic_range': 12}
        elif model == 'Phantom':
            self.params = {'camera_name': 'Phantom'}
            ## todo implement looking for parameters in a cine file
        elif model == 'QE_Pixelfly':
            self.params = {'camera_name': 'QE Pixelfly', 'nx_pixels': 1392,
                           'ny_pixels': 1024, 'pixel_xsize': 6.45e-4,
                           'pixel_ysize': 6.45e-4, 'quantum_efficiency': 0.62,
                           'f_analog_digital': 3.8, 'dynamic_range': 12}
        else:
            print('Camera not defined')
