import LibFILDSIM as ssfildsim
import LibParameters as sspar
import LibMap as ssmap
import LibPlotting as ssplt
import LibTimeTraces as sstt
import LibVideoFiles as ssvid

# Guess the machine to know which module we need to import
import os
a = os.path.abspath(os.getcwd())
b = a.split(sep='/')
for name in b:
    if name == 'ipp-garching.mpg.de' or name == 'ipp':
        import LibDataAUG as ssdat
        machine = 'AUG'

# Delte the intermedite variables to 'clean'
del a
del b
del os
del name
