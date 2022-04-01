"""Graphical user interfaces for the Scintilllator Suite"""

from Lib.GUIs.TomographyPlayer import ApplicationShowTomography
from Lib.GUIs.VideoPlayer import ApplicationShowVid
from Lib.GUIs.VideoPlusRemapPlayer import ApplicationShowVidRemap
from Lib.GUIs._RemapAnalyser import ApplicationRemapAnalysis
from Lib.GUIs._RemapAnalyser2D import ApplicationRemap2DAnalyser
# from Lib.GUIs.VideoPlusRemapPlayerProfiles import ApplicationShowProfiles
import Lib.LibMachine as m
machine = m.machine
if machine == 'AUG':
    from Lib.GUIs.BEP_gui import AppBEP_plot as ApplicationShowBEP
