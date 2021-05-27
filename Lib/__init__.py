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
import Lib.iHIBP as ihibp
import Lib.GUIs as GUI
import Lib.INPA as inpa
import Lib.LibOptics as optics
import Lib.BEP as libbep
import Lib.LibNoise as noise
from Lib.version_suite import version
import Lib.LibData as dat


machine = m.machine
paths = p.Path(machine)
# Delte the intermedite variables to 'clean'
del p
del m
# -----------------------------------------------------------------------------
# --- PRINT SUITE VERSION
# -----------------------------------------------------------------------------
print('-... .. . -. ...- . -. .. -.. ---')
print('VERSION: ' + version)
print('.-- . .-.. .-.. -.-. --- -- .')
