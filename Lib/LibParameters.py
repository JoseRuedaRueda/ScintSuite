"""@package LibParameters
Module containing physical constants and camera information

Basically here are hard-cored almost all the parameters of the suite
"""
# Physics constants
ec = 1.602e-19  # Electron charge, in C
mp = 938.272e6  # Mass of the proton, in eV/c^2
c = 3e8         # Speed of light in m/s

class Camera:
    """
    Class containing the properties of the cameras
    """
    def __init__(self, model):
        if model == 'VGA_Pixelfly':
            ## todo find the units in which pixel size ie given
            self.camera_name = 'VGA Pixelfly'
            self.nx_pixels = 640
            self.ny_pixels = 480
            self.pixel_xsize = 9.9e-4
            self.pixel_ysize = 9.9e-4
            self.quantum_efficiency = 0.40
            self.f_analog_digital = 6.5
            self.dynamic_range = 12
        elif model == 'Phantom':
            self.camera_name = 'Phantom'
            ## todo implement looking for parameters in a cine file
        elif model == 'QE_Pixelfly':
            self.camera_name = 'QE Pixelfly'
            self.nx_pixels = 1392
            self.ny_pixels = 1024
            self.pixel_xsize = 6.45e-4
            self.pixel_ysize = 6.45e-4
            self.quantum_efficiency = 0.62
            self.f_analog_digital = 3.8
            self.dynamic_range = 12
        else:
            print('Camera not defined')

