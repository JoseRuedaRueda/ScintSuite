"""
Install via pip the packages needed to run the suite

To be run in the main folder of the suite

Jose Rueda: jrrueda@us.es
"""
import os

# -----------------------------------------------------------------------------
# %% Create folders and files
# -----------------------------------------------------------------------------
cdir = os.getcwd()
pat = os.path.join(cdir, 'Data', 'MyData')
if not os.path.isdir(pat):
    os.mkdir(pat)
files = ['IgnoreWarnings.txt', 'Paths.txt', 'plotting_default_param.cfg']
for f in files:
    file = os.path.join(cdir, 'Data', 'MyData', f)
    template = os.path.join(cdir, 'Data', 'MyDataTemplates', f)
    if not os.path.isfile(file):
        os.system('cp %s %s' % (template, file))


