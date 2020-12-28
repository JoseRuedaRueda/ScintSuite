import LibFILDSIM as fildsim
import LibParameters as par
import LibMap as mapping
import LibPlotting as plt
import LibTimeTraces as tt
import LibVideoFiles as vid
import LibExtra as extra
import LibPaths as p
import LibMachine as m
import LibIHIBP as ihibpsim
from version_suite import version
if m.machine == 'AUG':
    import LibDataAUG as dat


machine = m.machine
paths = p.Path(machine)
# Delte the intermedite variables to 'clean'
del p
del m
