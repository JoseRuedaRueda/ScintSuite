# SCINTILLATOR SUITE

This code is created to analyze the signal from scintillator diagnostics. Support for FILD, INPA and i-HIBP is planned. To include a different diagnostic, please contact Jose Rueda: jose.rueda@ipp.mpg.de

## Installation and documentation
### Prerequisites. Python 3.7 or higher
Needed packages. Only listed *non-standard* packages. See below, there is a script to install them.

#### Essentials
The suite will not work without them:
- f90nml: To read FORTRAN namelist in an easy way (needed since version 0.1.10) `pip install f90nml`. This is the suite standard to read and write namelists!!!
- xarray: To handle the data sets (videos, remaps, timetraces, needed since version 1.0.0) [tested with version 0.20.1]
- numpy > 1.21.0: To support xarray

#### Optional (the suite will work but some capabilities will not be available)
- cv2 (OpenCv): To load frames from .mp4 files and to export videos `pip install opencv-python`
- lmfit: To perform the fits needed for the resolution calculation `pip install lmfit`
- cycler: To set default plotting color `pip install cycler`
- pyfftw: To have extra capabilities in the spectrogram calculation `pip install pyfftw`
- scipy 1.7.0 or newer, to have the RBF interpolators for the strike points
- aug_sfutils > 0.7.0: To load the AUG magnetic field (see AUG python documentation for the different ways of installing this package https://www.aug.ipp.mpg.de/aug/manuals/aug_sfutils/)
- mesh: To deal with CAD files
- numba > 0.55.1 to perform fast iHIBPsim xsection calculations and tomography
- odfpy: To read INPA logbook (which is written in an .ODS file)
For MAST-U users:
- pyEquilibrium:
  git clone https://git.ccfe.ac.uk/SOL_Transport/pyEquilibrium.git -b lkogan_aeqdsk pyEquilibrium
  pip install –user ./pyEquilibrium



#### Complete list:
In a clean-typical python installation with anaconda, taking care only of the packages and versions listed above should be enough and the suite will run smoothly, but python package dependence can sometimes a mess. As an indication, in the folder `Data/TestedEnv` you can find the result of the command `pip list` in a python environment where the suite was tested and working fine. So if you find a series of problems with packages versions, try to create your virtual environment and reproduce the installed package list detailed there. The files are labeled with the Suite version for which they were tested and 'Optx', meaning 'Option x', as different user can have different list of packages which could work.

### Cloning the suite and installing

The code can be installed via pip, just change `<branch>` for the name of the branch you want to install and <ParentFolderForSuite>

For AUG users: <ParentFolderForSuite> = `/shares/departments/AUG/users/$USER`
```bash
cd <ParentFolderForSuite>
git clone https://gitlab.mpcdf.mpg.de/ruejo/scintsuite ScintSuite
cd ScintSuite
git checkout <branch>
python first_run_pip.py
pip install -e .
```
This will install all requirement via pip. It is needed that your machine support pip installation of python packages

> IMPORTANT: If you install the suite outside your home dir, you should create an ennviromental variable pointing towards the suite folder. Add to your bash (or similar file) the following:
```bash
export ScintSuitePath=<ParentFolderForSuite>/ScintSuite
```
for AUG people, if they followed the recomended route for the code:
```bash
export ScintSuitePath=/shares/departments/AUG/users/$USER/ScintSuite
```


#### Advanced installation
Things can go wrong if your system has some particular rights limitations and you are not allowed to change them using `pip install`. In these cases, you should use a virtual environment:

1. Install virtualenv: `pip install virtualenv`
2. Create your virtual environment (let us call it SSvirtualenv): `virtualenv -p python3 --system-site-packages SSvirtualenv` [Note: for MU users, this has to be done as `python -m virtualenv ...`]
3. Activate your virtual environment (remember to do this everytime you are using ScintSuite or add it to your login script): `source SSvirtualenv/bin/activate`
4. Force install the compatible versions using `pip install modulename==X.X.X`. A list of compatible versions is listed here (checked in MAST-U and JET):
```python
scipy==1.7.0
scikit-image==0.16.2
pyfftw==0.12.0
pandas==1.3.1
```
For MU, you will probably also need:
`openpyxl==3.0.3`
`pco-tools`

Once the python modules are created, you need to create the folder `MyData` inside the Data folder, and copy in it the .txt files which are located in `Data/MyDataTemplates`. These are the configuration files of the Suite, they are needed to import the sutie and can be modified (the ones in MYData folder) to change the behaviour of the plotting, warning, paths... If you installed the sutie with the script: `first_run.py` this step was done already, so you can ignore it

### Getting started
**Importing the suite**

*Short story*: 
- DevUser (install method 1): Go to the main directory of the suite in your python terminal and run: `import ScintSuite as ss` (or change ss by the name you want)
- VainillaUser (install method 2): run `import ScintSuite as ss` from whatever place, as this is already in your python path

*Long story*: (only for method 1 of installation) In order to import the ScintSuite as `import ScintSuite as MyAwesomeName`, you need to set in your environment the different paths to the external modules. For example, in the case of AUG, the path towards the AUG-python library. To do this, you just need to run the file path suite. For example, just type in a python terminal `run paths_suite` (being on the main Suite directory). After running it, you should be able to import the suite from everywhere in your computer. However, if your working directory is the root directory of the Suite, there is no need of running this line, you can just execute directly `import ScintSuite as MyAwesomeName` and enjoy (as the function path_suite is called in the Sutie `__init__.py`)

**Using it**

Please see the examples in the Examples folder of the Suite, it contains the basic lines to execute the suite for each of they main capabilities. Please note than the examples does not contain all possibles ways of doing things iside the code, you will need to dig arround a bit if you need something too specific.

### Paths
There are three files containing the paths and routes for the suite:
- `paths_suite.py`: Located in the main directory of the suite. This one just contain the routes pointing to the python packages/files needed by the suite. It says to python where to find what it needs. If you need to add something due to the peculiarities of your system, please do it locally in your bash file or open a issue in gitlab, do not modify this file.
- `LibPaths.py`: Located inside the Lib folder, there they are all the paths pointing towards the different codes (FILDSIM, INPASIM etc) and the results/strike maps directories. Again do not modify this file just to put your custom paths, please.
- `MyData/Paths.txt`: It could happen that you have FILDSIM, SINPA or whatever installed in a route which is not *the official*. Inside this file, you can set all these paths. The file should be created when you run the script `first_run.py`; although you can always copy it from the `Data/MyDataTemplates folder`.

**Important:** If the ScintSuite is not installed in your home directory, you need to define the EnvVar `ScintSuitePath` pointing towards the installation location of the code. If that variable is not defined, the code will assume it is located in your home dir. 

VRT related paths are hardcoded. There is a significant number of them and overloading LibPaths seems like a poor solution. Blame Javier Hidalgo for this (jhsalaverri@us.es)

### FILDSIM notes
- You need to create an empty folder in the root of FILDSIM code with name 'cfg_files' in order to run the remap routine. The namelist of the new calculated strike maps will be stored here, so we do not create thousands of files in the main FILDSIM paths
- FILDSIM code receive no more support since version 0.8.0. FILDSIM libraries will not be updated further, except some important bug is found. Please use the new code version (uFILDSIM/SINPA)

### Documentation
- All objects and methods are documented such that the user can understand what is going on
- As everything has doc-strings, you can always write in the python terminal <fname>? and you will get all the description of the <fname> method or object
- The routines in the Example folder are intended to illustrate the use of the different tools in the suite. Please, if you want to play with them, make your own copy on 'MyRoutines', modifying the examples can cause merge conflicts in the future
- If you have installed Doxygen you can generate the documentation in html and LaTex format just opening a terminal in the Suite root directory and typing  `doxygen Doxyfile`. Once the documentation is generated, you can open the index with the following command `xdg-open doc/index.html`. For a (old and outdated) Doxygen generated documentation, see: <https://hdvirtual.us.es/discovirt/index.php/s/FBjZ9FPfjjwMDS2> download the content and open the index.html file, inside the html folder.

## Data export
All data exported and saved by the Suite is done in netCDF, as default format. Platform independendent and binary format.

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
