"""
i-HIBPgui to plot the videos of iHIBP.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import os
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import Lib
from Lib.SimulationCodes.iHIBPsim.crossSections import alkMasses
from Lib.SimulationCodes.iHIBPsim.geom import gaussian_beam
from skimage import measure

class app_ihibp_vid:
    """
    Class containing the data to create a GUI to plot the iHIBP videos and
    overplot the strikeline, when computed.
    """

    def __init__(self, tkwindow, shotnumber: int, path: str=None):
        """
        Initializes the class with the neccessary data to create the GUI and
        smooth plot the i-HIBP videos.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param tkwindow: window handler.
        @param shotnumber: shotnumber of the video to load.
        @param path: path to the simulation with iHIBPsim to plot the lines.
        """


        self.TKwindow   = tkwindow
        self.shotnumber = shotnumber

        # Loading the video object.
        self.video =

