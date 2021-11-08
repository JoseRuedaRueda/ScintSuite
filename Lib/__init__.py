"""
Main Core of the Suite

Allows to access all the diferent libraries and capabilities

Please see https://gitlab.mpcdf.mpg.de/ruejo/scintsuite for a full
documentation
"""
# Mapping
import Lib.LibMap as mapping

# Simulations codes
import Lib.SimulationCodes.FILDSIM as fildsim
import Lib.SimulationCodes.FIDASIM as fidasim
import Lib.SimulationCodes.SINPA as sinpa

# Reconstructions
import Lib.LibTomography as tomo

# Load data
import Lib.LibData as dat
import Lib.LibVideoFiles as vid


import Lib.LibParameters as par
import Lib.LibPlotting as plt
import Lib.LibTimeTraces as tt
import Lib.LibUtilities as extra
import Lib.LibFrequencyAnalysis as ftt
import Lib.LibPaths as p
import Lib.LibMachine as m
import Lib.LibTracker as tracker
import Lib.LibIO as io
import Lib.LibFastChannel as fc
import Lib.LibScintillatorCharacterization as scintcharact
import Lib.GUIs as GUI
import Lib.LibOptics as optics
import Lib.LibNoise as noise
from Lib.version_suite import version
import Lib.LibCAD as cad
import Lib.LibSideFunctions as side

machine = m.machine
paths = p.Path(machine)

# Non tokamak independent machines
if machine == 'AUG':
    import Lib.BEP as libbep
    import Lib.SimulationCodes.iHIBPsim as ihibp

# Delte the intermedite variables to 'clean'
del p
del m
# -----------------------------------------------------------------------------
# --- PRINT SUITE VERSION
# -----------------------------------------------------------------------------
print('-... .. . -. ...- . -. .. -.. ---')
print('VERSION: ' + version)
print('.-- . .-.. .-.. -.-. --- -- .')

# -----------------------------------------------------------------------------
# --- Initialise plotting options
# -----------------------------------------------------------------------------
# It seems that with some instalations, this could fail, so let's make just a
# try
try:
    plt.plotSettings()
except:
    print('It was not possible to initialise the plotting settings')
