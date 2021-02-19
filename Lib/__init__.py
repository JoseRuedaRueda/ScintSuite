import LibFILDSIM as fildsim
import LibParameters as par
import LibMap as mapping
import LibPlotting as plt
import LibTimeTraces as tt
import LibVideoFiles as vid
import LibUtilities as utilities
import LibPaths as p
import LibMachine as m
import LibTracker as tracker
import LibIO as io
import LibFastChannel as fc
import LibTomography as tomo

import iHIBP as ihibp
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
