import LibFILDSIM as fildsim
import LibParameters as par
import LibMap as mapping
import LibPlotting as plt
import LibTimeTraces as tt
import LibVideoFiles as vid
import LibExtra as extra
import LibPaths as p
# Guess the machine to know which module we need to import
import os
a = os.path.abspath(os.getcwd())
b = a.split(sep='/')
for name in b:
    if name == 'ipp-garching.mpg.de' or name == 'ipp':
        import LibDataAUG as ssdat
        machine = 'AUG'

paths = p.Path(machine)
# Delte the intermedite variables to 'clean'
del a
del b
del os
del name
del p
