# SCINTILLATOR SUITE

This code is created to analyze the signal from scintillator diagnostics. Native support for FILD, INPA. for iHIBP, you need the extra module of iHIBPsim. To include a different diagnostic, please contact Jose Rueda: jruedaru@uci.edu

## Citation of the code
To cite the code please use the ScintSuite [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13886726.svg)](https://doi.org/10.5281/zenodo.13886726)
## Installation and documentation
### Prerequisites
- Python 3.9 or higher
- Pip (anaconda is recommended)

### Cloning the suite and installing

The code can be installed via pip, just change `<branch>` for the name of the branch you want to install and `<ParentFolderForSuite>`

For AUG users is recommended: `<ParentFolderForSuite> = /shares/departments/AUG/users/$USER`
```bash
cd <ParentFolderForSuite>
git clone https://github.com/JoseRuedaRueda/ScintSuite ScintSuite
cd ScintSuite
git checkout <branch>
python first_run.py
pip install -e .
```
This will install all requirements via pip. It is needed that your machine support pip installation of python packages

> IMPORTANT: 
> If you install the suite outside your home dir, you should create an environmental variable pointing towards the suite folder. Add to your bash (or similar file) the following:
```bash
export ScintSuitePath=<ParentFolderForSuite>/ScintSuite
```
For AUG people, if they followed the recommended route for the code:
```bash
export ScintSuitePath=/shares/departments/AUG/users/$USER/ScintSuite
```
> IMPORTANT: 
> Each kind of terminal has its own command to define environmental variables. Please change the previous line as needed for your system.


#### Advanced installation
Things can go wrong if your system has some particular rights limitations and you are not allowed to change them using `pip install`. In these cases, a virtual environment could help:

1. Install virtualenv: `pip install virtualenv`
2. Create your virtual environment (let us call it SSvirtualenv): `virtualenv -p python3 --system-site-packages SSvirtualenv` [Note: for MU users, this has to be done as `python -m virtualenv ...`]
3. Activate your virtual environment (remember to do this every time you are using ScintSuite or add it to your login script): `source SSvirtualenv/bin/activate`
4. Force install the compatible versions using `pip install modulename==X.X.X`. A list of compatible versions is listed in Data/TestedEnv. There you can find the result of the command `pip list` in a python environment where the suite was tested and working fine. The files are labeled with the Suite version for which they were tested and 'Optx', meaning 'Option x', as different user can have different list of packages which could work.

Once this is done, run the script `first_run.py` to create the folder `MyData` and the settings file `settings.yml`. This are the configuration files of the Suite, they are needed to import the Suite and can be modified (the one in the root folder only!!!) to change the behavior of the plotting, warning, paths... If this script fails, you can copy the settings template manually. The templates is in the folder: `Data/MyDataTemplates`

### Getting started
**Importing the suite**

Except in the parent folder where the code is installed, just run:
```python
import ScintSuite as ss
``` 

**Using it**

Please see the examples in the Examples folder of the Suite, it contains the basic lines to execute the suite for each of they main capabilities. Please note than the examples does not contain all possibles ways of doing things inside the code, you will need to dig around a bit if you need something too specific. **Before doing anything new, please ask in the discord server, because most probably one of the other users already did it**

### Paths
There custom path to the different codes and folders must be added to the `Settings.yml` in the root folder of the suite. this file will be created when using the installation script. An example of the path section is:

```yaml
UserPaths:
  # Paths to the different simulation codes
  SINPA: '/pth/to/SINPA'

  # Paths to the remap databases
  StrikeMapDatabase:
    FILD: '/path to my strike maps database/'
    INPA: '/path to my INPA strike maps database/s'
    iHIBP: ''

  # Path to the scintillator plates
  ScintPlates: '/Path to where I store my plates'

  # Default paths to export/load results
  Results:
    default: '/path where I want all results to be exported by default'
    INPA: 'path where I want INPA results to be exported'
    FILD: '/path where I want FILD results to be exported'
    iHIBP: '/path where I want FILD results to be exported'
```


**Important:** If the ScintSuite is not installed in your home directory, you need to define the EnvVar `ScintSuitePath` pointing towards the installation location of the code. If that variable is not defined, the code will assume it is located in your home dir. 

### FILDSIM notes (legacy)
- You need to create an empty folder in the root of FILDSIM code with name 'cfg_files' in order to run the remap routine. The namelist of the new calculated strike maps will be stored here, so we do not create thousands of files in the main FILDSIM paths
- FILDSIM code receive no more support since version 0.8.0. FILDSIM libraries will not be updated further, except some important bug is found. Please use the new code version (uFILDSIM/SINPA)

### Documentation
- All objects and methods are documented such that the user can understand what is going on. NumPuy doc string is assumed. All new code since version 1.4.0 should be in this format
- As everything has doc-strings, you can always write in the python terminal <fname>? and you will get all the description of the <fname> method or object
- The routines in the Example folder are intended to illustrate the use of the different tools in the suite. Please, if you want to play with them, make your own copy on 'MyRoutines', modifying the examples can cause merge conflicts in the future


## Data export
All data exported and saved by the Suite is done in netCDF (h5), as default format. Platform independendent and binary format.

If the user is *alergic* to the use of programing languages in order to read the netCDF, this NASA program could be usefull: https://www.giss.nasa.gov/tools/panoply/download/ It allows you to open and plot the variables in the netCDF file


## Active Development
### Version control
Each release will be denoted by 3 numbers: a.b.c meaning:
- c: bug fixed and improved comments and documentation. Some new capabilities could be added (see changelog). The higher the number, the better.
- b: Significant changes, versions a.b1.c1 and a.b2.c2, should run perfectly with the same inputs.  But some internal routines may have changed, so if you have your own scripts using them 'outside the main loop' something can go wrong for you. The higher b, the more extra capabilities you have
- a: indicate major changes in the code, versions with different 'a' may be not compatible, not recommended update to a higher 'a' version close to a conference

### Branches
- master: Stable branch, things should work, may be a delay including new features
- dev-branch: developers branch, may have some small bugs or not fully developed features. Include the latest features, not recommended for general public
- 'tmp'-branch: linked to specific commits to include new features. Do not use these branches except you are the developer in charge of the new feature. Unicorns can appear

### Note for developers
- Before changing anything in a module open a issue in GitLab to start a discussion
- Indentation must be done via 4 spaces!
- PEP 8 guide is recommended, if some piece of code want to be merged without this standard, the maintainers could modify your code to adapt it to this standard (or completely deny your merge request)
  + maximum 80 character-long lines
  + space separation between operators, i.e., =a + b=
  + no blanks at the end of the lines
  + PEP8 in atom: <https://atom.io/packages/linter-python-pep8>
  + PEP8 in spyder: Tools > Preferences > Completion and linting > Code style and activating the option called #Enable code style linting#

### Issues and new implementations
If you are going to report a bug (or issue) please follow the template in <https://gitlab.mpcdf.mpg.de/ruejo/scintsuite/-/issues/71>

If a new implementation is required, open the appropriate issue in the GIT and link it to the milestone it corresponds (if possible). The following tags are available:

- Documentation: improve the documentation of a given section.
- Feature request: request to implement a new feature in the code.
- Minor mod.: request to implement minor modifications in the code.
- Enhancement: modify the implementation of a given feature to improve the efficiency or make easier some processing.
- Discussion: a forum to discuss ideas of implementation.
- Bug: minor error found in the code. To be corrected at the earliest convenience.
- Major error: an important error has to be solved in the code as soon as possible.
- Minor priority: Label for maintainer, indicates that the request has low priority in the ToDo list

## Machine names
All devices are identified by a string:
- `AUG`: ASDEX Upgrade
- `MU`: MAST Upgrade

## Useful links
- FILDSIM code: <https://gitlab.mpcdf.mpg.de/jgq/FILDSIM.git>
- SINPA (uFILDSIM) code: <https://gitlab.mpcdf.mpg.de/ruejo/SINPA>
- i-HIBPSIM code: <https://gitlab.mpcdf.mpg.de/poyo/ihibpsim>
- SMap library: <https://datashare.mpcdf.mpg.de/s/yyLR7hCKNBqK34W>
- Phase correction for magnetics: <https://datashare.mpcdf.mpg.de/s/FiqRIixNMb82HTq>

## Implementation of other machines
The suite is thought to be machine independent, but some work must be done:
- Create a module equivalent to LibDataAUG with the database methods of your machine
- Include your paths in paths_suite.py and LibPaths.py
- Include the calling of your nice module in LibMachine
- Cry a bit because some thing might still don't work
- Send an e-mail to jrrueda@us.es (maybe also some chocolate?)
- Wait a couple of days for him to solve the issues
- Enjoy!
