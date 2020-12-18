import os
a = os.path.abspath(os.getcwd())
b = a.split(sep='/')
for name in b:
    if name == 'ipp-garching.mpg.de' or name == 'ipp':
        machine = 'AUG'
del a
del b
del os
del name
