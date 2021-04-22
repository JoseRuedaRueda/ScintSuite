import LibFILDSIM as fildsim
import LibParameters as par
import LibMap as mapping
import LibPlotting as plt
import LibTimeTraces as tt
import LibVideoFiles as vid
import LibUtilities as extra
import LibFrequencyAnalysis as ftt
import LibPaths as p
import LibMachine as m
import LibTracker as tracker
import LibIO as io
import LibFastChannel as fc
import LibTomography as tomo
import iHIBP as ihibp
import GUIs as GUI
import INPA as inpa
import LibOptics as optics
from version_suite import version
if m.machine == 'AUG':
    import LibDataAUG as dat


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
