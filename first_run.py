"""
Install via pip the packages needed to run the suite

To be run in the main folder of the suite
"""
import os
import pip


def import_or_install(package, name=None):
    """Import or install a package."""
    try:
        if name is None:
            __import__(package)
        else:
            __import__(name)
    except ImportError or ModuleNotFoundError:
        print(package, 'not found')
        i = input('1 to install, otherwhise skip')
        if int(i) == 1:
            pip.main(['install', package])


# -----------------------------------------------------------------------------
# Create folders and files
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

# -----------------------------------------------------------------------------
# --- Look for all suite packages
# -----------------------------------------------------------------------------
print('MACHINE DEPENDENT LIBRARIES AS AUG_SFUTILS MUST BE INSTALLED INDEPENDENTLY')
import_or_install('f90nml')
import_or_install('opencv-python', 'cv2')
import_or_install('lmfit')
import_or_install('cycler')
import_or_install('pyfftw')
import_or_install('netCDF4')
import_or_install('shapely')
import_or_install('stl')
import_or_install('pco-tools')
import_or_install('xarray')
import_or_install('tqdm')
import_or_install('matplotlib')
import_or_install('numpy')
import_or_install('numba')
import_or_install('scipy')
import_or_install('scikit-image', 'skimage')
import_or_install('scikit-learn', 'sklearn')
