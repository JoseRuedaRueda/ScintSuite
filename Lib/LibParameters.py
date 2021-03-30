"""Contain physical constants and camera information

Basically here are hard-cored almost all the parameters of the suite
"""
# Physics constants
ec = 1.602176487e-19  # Electron charge, in C [from NIST]
mp = 938.272e6  # Mass of the proton, in eV/c^2
mp_kg = 1.67262192369e-27  # Mass of the proton in kg
c = 2.99792458e8       # Speed of light in m/s
amu2kg = 1.660538782e-27  # Scaling factor to go from AMU to SI units (NIST)
h_planck = 4.135667e-15         # [eV/s]


iHIBP = {'port_center': [0.687, -3.454, 0.03], 'sector': 13,
         'beta_std': 4.0, 'theta_std': 0.0, 'source_radius': 7.0e-3}


# -----------------------------------------------------------------------------
# --- File parameters
# -----------------------------------------------------------------------------
filetypes = [('netCDF files', '*.nc'),
             ('ASCII files', '*.txt'),
             ('cine files', ('*.cin', '*.cine')),
             ('Python files', '*.py')]

# Access to files via seek routine.
SEEK_BOF = 0
SEEK_CUR = 1
SEEK_EOF = 2
SEEK_END = 2


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
            print('Most Phantom parameters are written in the .cin file')
            print('Just open the Video object and enjoy')
        elif model == 'QE_Pixelfly':
            self.params = {'camera_name': 'QE Pixelfly', 'nx_pixels': 1392,
                           'ny_pixels': 1024, 'pixel_xsize': 6.45e-4,
                           'pixel_ysize': 6.45e-4, 'quantum_efficiency': 0.62,
                           'f_analog_digital': 3.8, 'dynamic_range': 12}
        else:
            print('Camera not defined')
