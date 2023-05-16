"""
Installer file for the package.
"""

from setuptools import setup

import sys
import os

libpath = os.path.join(os.dirname(__file__), 'Lib')
sys.path.append(libpath)

from Lib.version_suite import version

# Loading the dependencies from the requirements file.
install_requires = list()
with open('.setup/requirements.txt', 'rt') as fh:
    for line in fh:
        install_requires.append(line.strip())
    
optional_dependencies = list()
with open('.setup/optional_requirements.txt', 'rt') as fh:
    for line in fh:
        optional_dependencies.append(line.strip())

# Loading the long description from the .setup/README.md file.
with open('.setup/description.txt', 'rt') as fh:
    short_desc = fh.readline()
    long_desc = fh.read()

# Loading author list from the .setup/authors.txt file.
with open('.setup/authors.txt', 'rt') as fh:
    author_list = list()
    email_list  = list()
    for line in fh:
        author, email = line.split('-')
        author_list.append(author.strip())
        email_list.append(email.strip())

# Reading the keywords from the file.
with open('.setup/keywords.txt', 'rt') as fh:
    keywords = [line.strip() for line in fh]


setup(
    name='ScintSuite',
    version=version,
    descritpion=short_desc,
    long_description=long_desc,
    author = author_list,
    author_email = email_list,
    url = 'https://gitlab.mpcdf.mpg.de/ruejo/scintsuite',
    maintainer=author_list[0],
    maintainer_email=email_list[0],
    license='GPLv3',
    classifiers=keywords,
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
    extras_require=optional_dependencies,
    python_requires='>=3.7'
)


import shutil
# We will now configure the system for the user.
shutil.copyfile('Data/MyDataTemplates/Paths.txt', 'Data/MyData/Paths.txt')
shutil.copyfile('Data/MyDataTemplates/IgnoreWarnings.txt', 'Data/MyData/IgnoreWarnings.txt')
shutil.copyfile('Data/MyDataTemplates/plotting_default_param.cfg', 'plotting_default_param.cfg')