"""
Install via pip the packages needed to run the suite
"""

import pip


def import_or_install(package):
    """
    Import or install a package
    """
    try:
        __import__(package)
    except ImportError:
        print(package, 'not found')
        i = input('1 to install, otherwhise skip')
        if int(i) == 1:
            pip.main(['install', package])


# -----------------------------------------------------------------------------
# --- Look for all suite packages
# -----------------------------------------------------------------------------
print('MACHINE DEPENDENT LIBRARIES AS AUG_SFUTILS MUST BE INSTALLED INDEPENDENTLY')
import_or_install('f90nml')
import_or_install('cv2')
import_or_install('lmfit')
import_or_install('cycler')
import_or_install('pyfft')
import_or_install('netCDF4')
