"""
Main Core of the Suite

Allows to access all the diferent libraries and capabilities

Please see https://gitlab.mpcdf.mpg.de/ruejo/scintsuite for a full
documentation
"""
import Lib.LibFIDASIM as fidasim
import Lib.LibFILDSIM as fildsim
import Lib.LibParameters as par
import Lib.LibMap as mapping
import Lib.LibPlotting as plt
import Lib.LibTimeTraces as tt
import Lib.LibVideoFiles as vid
import Lib.LibUtilities as extra
import Lib.LibFrequencyAnalysis as ftt
import Lib.LibPaths as p
import Lib.LibMachine as m
import Lib.LibTracker as tracker
import Lib.LibIO as io
import Lib.LibFastChannel as fc
import Lib.LibTomography as tomo
import Lib.LibScintillatorCharacterization as scintcharact
import Lib.GUIs as GUI
import Lib.SINPA as sinpa
import Lib.LibOptics as optics
import Lib.LibNoise as noise
from Lib.version_suite import version
import Lib.LibData as dat
import Lib.LibCAD as cad
import Lib.LibSideFunctions as side

machine = m.machine
paths = p.Path(machine)

if machine == 'AUG':
    import Lib.BEP as libbep
    import Lib.iHIBP as ihibp

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
plt.plotSettings()
