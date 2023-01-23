# 1.2.1:
## StrikeMaps
- avg_ini_gyrophase, avg_ini_gyrophase, renamed to avgIniGyrophase, avgIncidentAngle, to avoid issues in the automatic remap
## Plotting
- IncludeColorMap flag added to multiline()
# 1.2.0: Melon con Jamon
## General
- Changed @ for : in documentations
## Data
- INPA object documentation improved
- getGeomShots() removed from INPA object
## Simulation codes
- Added compatibility with SINPA 4


# 1.1.0: Melon con Jamon
## FIDASIM library
- FIDASIM library distributed with the suite is no longer mantained. Please download it from the main repo of FIDASIM4
## Logging
- Logging output was colorised to distinguish between info and warning
## Scintillator
- New library combining scintillator efficiency and characterization
## Optics
- Output of the defocusing matrix is now a sparse matrix, to save memory (still not recommended to be used)
## Side Functions
- Included a function to generate a gaussian kernel for convolutions gkern
## Scintillator
- New library combining scintillator efficiency and characterization
## Strike Map Library
- Defocusing included in the calculation of the INPA instrument function via convolution
## Video
- corrected the reading of the shutter time in NS from png file, it was being stored in s
## Others
- Small typos corrected in documentation
- Updated readme
- git commit now shown in starting of the Suite and saved in the version file
# 1.0.4: Lentejas
## Tomography
- Lib Tomography rewritten, now the folding of the W is done via numba, saving 99% of the time
- New class to perform tomography created
- Tomography library split in different small files
## Optics
- Included a simple grid object, to generate evenly spaced grid (usefull for distortion). The object is a child of XYtoPixel
## SimulationCodes
- field.tofile() now accept a string as fid. If this is the case, the method will open itself the file, so there is no longer the need of open the file outside
## Video
- INPA video now have a flag to load NBI and ne data when reading the video header
- *BUG*, corrected a bug when substracting the noise in a video giving just the frame
- Added and optional argument, YOLO, in the BVO. If true, the code will ignore frames which database is corrupt, typical
## GUIs
- Adapted GUIs to the new xArray structures
- Created GUIS to show raw video and plasma traces

## 1.0.3: Salmorejo con Jamon
### TimeTrace
- Time trace object have now a method to export to netCDF
- Timetrace object allows now to be initialised from a netCDF file
- read_trace() from the io package was deprecated, as it used the old format for the trace
- TimeTrace accept now a name as input, to be used to label the plots

### Video
- FILD video includes now a scintillator attribute
- getTimeTrace return the time trace using the scintillator as mask, if no time nor mask is passed as input

## 1.0.2: Bug fix and example revision
- Examples revised by lvelarde
- **Bux fix** small bug relative to the us of the xarrays in the video object fixed
- Improvements in code documentation and comments
- FILD remap now is provided with units
- read_from_loaded was deprecated from the read_frame() method of the video, please use just getFrame()

## 1.0.1: Salmorejo con picatostes
- Minor bug fix in the GUI which plot the frames
- Integral of the remap is no automatically calculated in the remap video
- Minor bug fix and comments improvements

## 1.0.0: Salmorejo
**Installation instructions**: Please run the file 'first_run.py' if you just landed in version 1.0.0, as some extra files need to be created
### General comment:
- The disorder 'gyr', 'rl', 'Gyroradius' etc was eliminated. Now it is called 'gyroradius' at every point of the suite (please if you find any not update, tell jrrueda@us.es)
- Changelog and readme are now written in markdown format, which is more open and easy
- Lib name removed from files, to make things more readable. All submodules as _, to avoid duplicities in import (Lib data and Simulation codes will be addapted in the next version)
- All (almost) plotting function now return the axes where the data was plotted
- No longer support for AUG-dd libraries (this caused the magnetics and ECE routines to do not work, will be solved in future releases)
- calling to LibDat to get the IpBt sign as default argument in pitch functions is eliminated (it was creating an infinite import loop) Now all IpBt default to -1, and the user should take care of that sign when calling a pitch function
- Use of `logging` module to handle warnings and prints implemented. Up to now, just the basic warnings are inside this framework. minor prints will become logger info in future versions
- Now the remapping can be done in any arbitrary variable, such as energy pitch for FILD or Energy-rho for INPA. Notice that this causes problems with the 'translation remaps'. WIP

### LibCAD:
- write_file_for_fortran based on open3D was deprecated, as this package is no easy to install in cluster and non-personal computers
- write_file_for_fortran_numpymesh renamed as write_file_for_fortran and consider as the official function. A dummy wrapper with this name was left in the library for retro-compatibility. It will be removed in version 1.1.0
- open3D no longer used nor needed

### LibData:
- AUG `get_shot_basic()` was adapted to new sfaug library
- INPA PMT channels data was included

### GUIS
- StrikeMap button fixed, but only SINPA format is supported now (if users ask for it, FILDSIM format could be back, but the new smap archives are done with SINPA so...)

### LibMapping:
- The strike map object there present is just a wrapper for the new strike maps
- The remap done with a mask was improved to save memory (now just z values are set to zero instead of cutting the 3 arrays)
- The rprofmin and other variables to calculate profiles during the remap where eliminated, please use the integrate remap function if you want a profile

### LibOptics:
- Included methods to simulated finite focus in the cameras

### LibPlotting:
- Divided in submodules for easier iteration
- Cursors added for multi-axis analysis

### SimulationCodes.Strikes
- includeW no longer used in calculate histograms. Both histograms weight/no weights will be calculated (as just miliseconds are needed for the calculation and the histogram occupies quite small space in memory)
- Corrected the get() function such that it properly return a flattened array
- Included normalise flag in the histogram plotting routine

### LibStrikeMap:
- Created a proper object to handle the strike map. Now there is a parent general class from which each diagnostic feed. No longer messy naming and ifs inside the object to distinguish between diagnostics
- The StrikeMap object of the mapping library is just a wrapper to this new objects, to keep retro-compatibility
- Old criteria of x,y,z being y,z the scintillator and then x,y the scintillator plane in the camera was abandoned. Now the coordinates in the scintillator are called x1, x2, x3, being x1 and x2 in the scintillator plane and x3 normal to it. Camera coordinates are kept as x,y
- Many new side functions and property were added to the strike map, for example the plot_variable, to get a quick insight of a given variable and the properties such as size and variables, to see how many pairs of points contain the map or which variables are stored
- Some functions of the Strike map changed names:
    - calculate_resolutions -> calculate_phase_space_resolution
    - map_to_txt -> export_spatial_coordinates. map_to_txt remain there for compatibility as a wrapper, will be deleted in version 11.2
- Some where deprecated completely
    - get_energy()
    - plot_strike_points() (it had a deprecation warning since version 8, sot it was time to remove it)
- Some change some input naming, examples:
    - in plot_pix, plot_real, now the labels are rotation_for_x_label and rotation_for_x_label if you want to rotate the labels, x is for the first montercarlo variable, x for the second.

### LibTimeTraces
- **bug**: Corrected the bug where the plotting of a time trace with a baseline correction was overwritting the data in the timetrace_object
- Rewritten the object to use xarray as core. This open the door to share a basic object between the TT object and the fast channel one, simplifying the future

### LibTracker
- Finally removed, use directly iHIBPsim library

### LibVideo: Tiff Support
- Included full support for TIFF files. Now they can be used as input video format
- 'normalise' flags was added to the plot_remap_frame routine, to normalize the plot to unity
- **bug**: Solved a bug in the get frame index
- **bug**: Solved a small bug, the .cin files was not being closed in the read_settings structure
- Translate remap: temporally not available, until it is adapted to the new remap structure
- getTimeTrace now returns also the used mask
- exp_dat, remap_dat are now xarrays instead of dictionaries
- export_remaps now create a series of netCDF files to simplify reloading of data

### LibIO
- read_calibration() routine now compatible with distortion calibrations
- Included load_remap(), to load and handle the new remap exports

### LibFastChannel
- Was rewritten to derive from the common timetrace class

### LibMachine
- Made a bit more robust the case of false AUG detections, but still not good enough, we need a better way

### Others
- getGyroradius and getEnergy now uses the proper amu to kg conversion and no longer rely on the proton mass
- Integrate_remap change completely inputs and output, now it can handle also any translation of the remap. Please have a look at the new function doc. Take care with the translation if you do not use standard remaps as initial point
- The small bug on the units of the weights of INPA markers from SINPA was corrected (the correction due to the detector pinhole size was already performed in SINPA)
- Included new sub-library MHD, to calculate/plot basic mode parameters such as mode frequencies


## 0.9.10: iHIBP videos improvements and TORBEAM sims.
- Corrected bugs in the iHIBPsim videos read.
- Added a class to handle the TORBEAM simulations written in UFILE format.
- Changed ECRH time traces reading into SFUTILS library.

## 0.9.9: BEP removed from library and work in iHIBP.
- Removed the full BEP library from the ScintSuite.
- Re-branded class for the iHIBP video with:
    - Distorted and aligned scintillator.
    - Plotting time traces of the signal within the scintillator.
    - Basic and advanced methods to substract noise from the signal.
- Included in LibData/AUG/DiagParams the paths to the relevant iHIBP files.

## 0.9.8: Rebujito release
- Filtered Numpy Warnings
- Small PEP8 corrections
- Solved a bug in the fidasim.read.read_npa routine. kind of the markers was not being read
- New FBM and NPA objects to load and plot the FIDASIM FI distribution function and neutral CX flux
- FIDASIM library re-written
- Simplified internally the plot_frame from the vido object

## 0.9.7: Moving to SFUTILS.
- Changed dd-libraries in LibData/AUG/Misc into sfutils.

## 0.9.6: Energy remap
- Include the possibility to translate the remap to energy instead of rl (see the function translate_remap_to_energy of video object and the plot_frame remap for the needed input)
- FILDgeometry/INPAgeometry atributes in the video object just renamed to geometryID, idem with FILDposition, FILDorientation etc to simplify life
- Added a proper first_run script


## 0.9.5b Text and stl output functions added to evaluate SINPA data structures
### SimulationCodes Lib
 - Functions to write triangles to an STL file has been inlcuded in geometry
 - A function to write the strikepoints data obtained from a SINPA simulation to txt file has been added, to be able to easily load the data in CAD software

### LibMapping
 - Functions to write strike map points to a txt file has been added a swell


## 0.9.5fix: Bugfix for writing >2D fields to Fortran files.
### SimulationCodes/common/fields & SimulationCodes/iHIBPsim/profiles
Writing and reading from files using 2D arrays to Fortran programs was enabled via an _ad-hoc_ transpose. This method was creating a problem with 3D and above dimensions, since np.tofile does write in 'C'-order nowithstanding the np.asfortran array.
Found how to write a proper Fortran 3D array in <https://stackoverflow.com/questions/22385801/best-practices-with-reading-and-operating-on-fortran-ordered-arrays-with-numpy>


## 0.9.5: Small upgrade
### LibVideo
- Plot frame and plot remap frame include new optional arguments
- New BWR added in video player

### LibSimulationCodes
- Strikes objects upgraded for new wrong markers file in SINPA

### LibStrikeMap
- Small correction in the strike map plotting for INPA


## 0.9.4 iHIBPsim norm correction and SFUTILs in profiles.
### SimulationCodes Lib
- iHIBPsim has now a class handling the deposition generated by the iHIBPsim code allowing for direct plotting of the results -> Lib.ihibp.depos.deposition(...)
- Deposition and 1D strikeline histogram corrected with the proper normalization. The normalization allows comparing different number of particles, i.e., different resolutions.

### AUG/Profiles
- The main functions (get_ne, get_Te, ...) has been moved now to the SFUTILS library and work significantly faster.
- Still to change ECE routines into SFUTILS.
- Now the get_Ti_CXRS and get_tor_rotation_CXRS allow for a single time point to be provided instead of forcing a time window.


## 0.9.3: Strike modelling
### LibOptics
- Include a class to accommodate radial dependent transmission coefficient in the cameras

### LibVideoFiles
- Include a GUI to analyse the remap in 2D (allows to subtract a reference frame and plot timetraces)

### SimulationCodes
- Optical calibrations (magnification, transmission and distortion) can be applied now to the strike points

### Others:
- Improve documentation and error handling
- Small improvementes in INPA strikemap plotting


## 0.9.2 Added functions to the VRT library
- Added the get_time_traces in LibVRT
- Added the ROI2mask function to LibVRT
- Fixed a bug on the getdist2sep of FILD4


## 0.9.1 Minor bug fix and upgrades
### SimulationCodes Lib
- Now the strikes object will look backwards in the strike header file, ie, if the used SINPA version is X.y, and there is no specific header in the StrikeHeader file for version X, the code will assume that there was no changes in the file structure between version X and version X-1 and use the header of version X-1
- Added a function to read the fast-ion distribution function use in FIDASIM
- solved the bug in the FIDASIM library init file, which make the FIDASIM library to be loaded in loo

### FILD-INPA object
- function plot_orientation() renamed to plotBangles() to avoid confusion between the FILD orientation (alpha, beta, gamma defined in the machine system) with the magnetic field orientation (theta and phi)


## 0.9.0 INPA implementation
### General:
- Time traces library was re-written to do not depend on external libraries and avoid bugs. Now is also way faster for non-cine videos, as it was re-written to avoid loops
- run path_suite is no longer needed to import the Suite (see import section in the readme)
- Subfolders for each diagnostic were created in the folder Data/Plates

### INPA and Lib.Video:
- Included INPA calibration files in the data folder
- FILD INPA and Basic Video Object moved to _<name> just to clean a bit the vid object
- FILD and INPA do not depend directly from BVO but from the new object FIV, which contains the common rouitnes from FILD and INPA, as the itegral of the remap signal
- Included INPA paths
- Lib.LibVideo.FILDVideo.plot_frame() now include the flag IncludeColorbar to include or not the colorbar (default=True). Also, it includes the parameter alpha to have transparencies
- Lib.LibVideo.FILDVideo.plot_frame() now includes the time as text inside the box, not as the title
- Lib.LibVideo.FILDVideo.plot_frame() was moved to the new parent object
- Lib.LibVideo.FILDVideo.integrate_remap(), now included in the parent object, and return not only the trace but the marginal distributions in rl and pitch (of R for the INPA)

### Lib.SimulationsCodes:
- Same naming criteria was adopted for FILDSIM and SINPA, so the routines  guess_strike_map_name_FILD from the SINPA and FILDSIM libraries were renamed just into guess_strike_map_name

### Examples:
- Added Example 16 of the FILD collection to align the scintillator

### LibData:
- get_fast_channel adapted to use the aug_sfutils instead of the old dd

### LibMapping:
- Updating the Scintillator object to make it compatible with the SINPA format
- readCameraCalibration was extracted from the FILDlogbook and placed to the mapping library, as it will be used for INPA and FILD at the same time
- Calibration params now include 3 parameters to model distortion (distortion center xc and yc and distortion coefficient). Space for a 4th coefficient was allocated, although not needed for INPA dna iHIBP
- Scintillator, and strike maps now depend of the parent class XYtoPixel, which contain the basic information of cooridnates and pixel position and the method to translate among them

### Plotting:
- Updated plotSettings() to match new matplotlib. font_manager changed and latex preamble removed

### LibFastChannel:
- Default plotting option changed to raw, to do not fail if the user plot the data just after loading them, so no filtering was done

### Bug:
- Solved a bug in plot_frame from the FILD video object. The syntax for the routines to load the strike map was not updated to version 0.8.0
- Solved a bug in timetraces object. Mean and std of the ROI were exchanged
- Solved a bug in the StrikePoints.plot3d(). If the percentage was too low that no marker was selected. Now the code just check that some markers was actually selected

### Other changes
- Some comparison to strings changed to <strin>.lower() == ... to give more flexibility in case the user capitalise the first letter
- np.bool replaced with bool to avoid future issues with numpy (np.bool was going to be deprecated)
- PEP8 improvements
- Comments and documentation (Readme) improved


## 0.8.3 MAST-U adaptation:
In the process of adapting the code to work for the MAST-U FILD, some important changes have been performed:
- New library to read PCO files (format .b16)
- Bug fixed in PNG library: the video files were not necessarily read in the correct order


## 0.8.2 Added get_pellets_timeTrace and update LibFILD4
- Routine to get the pellets time trace
- Included a routine (get_dist2sep) to calculate the distance to the separatrix


## 0.8.1 FILD4 object added (LibFILD4)
- Added routines to load and reconstruct FILD4 trajectories as part of LibData
- FILD4 database is stored in Javier Hidalgo local machine. Contact him if you cannot access them.
- Routines used to load and plot FILD4 trajectories now show as deprecated


## 0.8.0 SINPA implementation data analysis
#Notice#: For all the SINPA related implementation, you need version 0.3 of the SINPA code
### Data Folder:
- calibration_database.txt was moved from cm (used by FILDSIM) to m (used by SINPA). A copy of the old file is kept, saved as calibration_database_cm.txt (see FILD example 0 of how to easily use this file)
- StrikeMaps will have to be now included inside the folder RemapStrikeMaps/FILD/<geomID>, where <geomID> is the geometry ID of the FILD head used

### Examples Folder:
- SINPA examples updated to the new namelist paraters
- FILD examples simplified thanks to the direct and easy way of handling now the video files
- L3 from FILD lectures replaced. There is no longer need for custom options for RFILD. Now L3 shows the new averaging capabilities
- L14 and L16 from FILD lectures was deprecated (as it was never complete neither machine independent).

### Lib.GUIs:
- VideoPlusRemapPlayer: Was addapted to the new strike map database structure (still missing some tweaks to be included in version 0.8.1)

### Lib.Data.Equilibrium:
- Moved to the aug_sfutils library to load the magnetic field. This library is faster. But you need version 0.7.0 or newer

### Lib.Data.FILD:
- Minor improvements in FILDlogbook

### Lib.Data.DiagParam:
- FILD6 (RFILD) was deleted from the parameter list. As agreed with Javi, RFILD will be just FILD1 with the geom AUG01, as it was in reality, same manipulator, same camera, same pmts...
- R,z, phi, alpha and beta were removed from the hardcored parameters. Now theses default parameters are defined via namelist in the data folder. For each FILD geometry (see logbook)

### Lib.Map.FILD
- Remap all FILD frames completely rewritten, removed unnecessary parameters/calls. Removed calling the magnetic field inside this function. This makes mode transparent and easy to make the code machine agnostic

### Lib.Map.StrikeMap
- #Bug Solved# Solved a bug which caused problems while calculating the resolutions for the cases where no strike points reach the scintillator for a given gyroradius or for a given pitch angle
- #Bug Solved# Solved the issue of data from different pitches values not being stored in the proper place of the strike map object.
- #Bug Solved# Solved issues in plot_resolution_fits, the variables index_pitch and index_gyr were float, so the code failed while using them as indeces (detected by Alex)

### LibVideo.AuxFunctions:
- The function guess_filename() from the auxiliary functions of the Video library was moved towards the LibData.AUG.FILD, because at the end this was using AUG criteria. This should simplify MAST-U implementation. Also, it was renamed to guessFILDfilename

### LibVideo.BasicVideoObject:
- flag 'empty' was included in the BVO such that the video object can be initialize empty. This is to initialize the video object from remap saved files
- The BVO includes the possibility to average the video on an arbitrary time base. These average frames can be used as input for the remaping routines

### LibVideo.FILDVideoObject
- Now fetch FILD position, orientation and geometry from the FILD logbook
- Now include the magnetic field as an attribute of the object to better handle the remap
- It can be initialized just with the shot number and the desired FILD ID
- export_remap() addapted to work with the new internal structure of the VideoObject
- remap_loaded_frames(): Changed completely to adapt to SINPA code and to be more machine independent. #INPUT changed#. Notice that now the code will identify by the namelist if it needs to launch SINPA or FILDSIM. If the strikeMap folder is 100% empty (not even the basic namelist) this will fail
- Use the flag use_average in the options dictionary in the remap input to use the experimental or average frames

### Lib.SimulationCodes.Common
- Geometry object has now a routine to generate files in SINPA format
- Plot2D with shaded areas included (thanks to @Alex)
- Function Strikes.calculate_2d_histogram and plot_histogram will calculate and plot all histograms you could imagine
- StrikeHeader from SINPA updated to match SINPA units (m)

### Lib.SimulationCodes.FILDSIM
- guess_strike_map_name_FILD change its optional arguments, now it is geomID, not machine, as FILD geometries are now identified by a geometry id
- run_FILDSIM has now an input named cluster, though for the future implementation of MAST-U clusters

### Lib.SimulationCodes.SINPA
- write_namelist() now also prepares the directories main, results and inputs, to simplify execution
- find_strike_map_FILD() created. This is equivalent to the one of the FILDSIM package, it try to find a strike map, if can not find it, it creates it
- #Bug Solved# Solved a bug in executing the SINPA code via SBATCH file (Thanks to @Alex)

### LibIO
- load_FILD_remap(). New function from the io library allows to load a remap file into a video object

### Lib.errors: Custom Exceptions
- Custom exceptions are here. They are defined in the file errors.py and are created to be more precise when the Suite raise and exception. This allows better filtering with try structures. Many of the raised exception are now handle by this way. The rest will come in the future

### DEPRECATED
- The Strikes object of the FILDSIM code, use the Common object instead, already available and working better. You can use it, but it would give you a warning
- StrikeMap.plot_strike_points() is deprecated. Please use StrikeMap.strike_points.scatter() instead, much better, with more flexibility and options

### Others
- np.arange substitute by range in loops
- Comments improved
- Small changing to correct deviations from PEP8
- Updated readme
- File First_run.py which only confused new users was removed
- Included an issue template


## 0.7.9 iHIBPsim updated.
- Minor errors corrected in the iHIBPsim libraries.
- iHIBPsim namelist: library ready to read and parse the namelists that will be used as inputs for the i-HIBPsim fortran code.
- iHIBPsim beam: the library has been updated and a simple GUI is introduced (Examples/Others/ihibpsim_beam_gui)
- iHIBPsim video viewer: included viewer in Examples/Others/ihibpsim_video_gui. No calibrations are yet applied.
- iHIBPsim paths updated in LibPath
- Optical calibration of the i-HIBPsim plate added (Data/Calibrations/iHIBP).

### Profile library in iHIBPsim.
- Profiles class to read from the database and save them for the iHIBPsim execution (SimulationCodes/iHIBPsim/profiles.py)
- Can read from the database.
- Save/read the binary files.
- Plotting routines.
- Possibility to modify the 1D profiles to study perturbations.


## 0.7.8 FILD logbook
- FILD loogbook object was upgraded. Now is a complete database to interact with the object
- The function to read the optical calibration database was moved into the FILD logbook object. The old one remains, but marked as deprecated
- Deprecated decorators where included in the suite (thanks to PLEQUE code :)
- #Note# This is a transitional update, in version 8.0 the FILD logbook will be directly use in the automatic remap


## 0.7.7 Small improvements in handling SINPA and FILDSIM
### Examples:
- SINPA examples were updated the new SINPA code version (which enables the default parameters in the namelist so FILDSIM user do not need to worry about INPA variables)

### Mapping:
- StrikeMap.calculate_resolutions and StrikeMap.remap_strike_points() where updated to ensure INPA compatibility

### SimulationCodes:
- Strikes object now have the method .get() which return the data from the desired variable of the strike points

### Others:
- Small improvements in comments


## 0.7.6 VRT video object and LibVRT
### VRTVideoObject
- Solved a bug where the time trace was not the same as in the loaded video

### LibVRT
- Library to interact with the VRT data
- Get camera calibration (signal -> temperature) and (some) camera configuration parameters


## 0.7.5 VRT video object and loadMask
### VRTVideoObject
- Object intended for the analysis of the VRT cameras. Children of the BasicVideoObject
- Can plot VRT videos and save ROIs

### LibIO
- Added load_mask


## 0.7.4 Massive remaps:
- A flag 'allIn' was included in the function to remap all loaded FILD frames. If this flag is set to true, the code will always take the closer strike maps, without allowing to the user to calculate the strike map. In this way, you can remap 'N' shot automatically, without having to say 'No' to the program if a strike map is missing
- #Bug_solved#. Bug which make the load of png files not possible is solved (the bug was introduced in version 0.7.0)


## 0.7.3 SINPA examples:
### Examples:
- Examples to execute the SINPA code polished and more documented.

### Bugs:
- Solved bugs when several smaps of SINPA where loaded, a dictionary was not been properly copied so problems appeared in the header
- Solved a bug in the Smap.plot_resolution_fits() routine, due to copy/paste, an index in the loop was ir instead of i
- Solved a bug in the SINPA init module, geometry module was not loaded properly
- Solved a small bug in the Smap.plot_resolutions(), the old convention 'pitch' instead of 'XI' was used there


## 0.7.2 Logbook:
### LibDat:
- A new FILD class was created. This class read directly the FILD logbook (excel on the web) and get the FILD position and orientation for that shot
- CalibrationDatabase.txt was moved into a folder AUG in the FILD folder inside the Calibration folder of the Data folder. This was made to accommodate future calibrations for other tokamaks
- Default_positions.txt was added in the FILD calibraation folder. The code will use the positions and orientation of FILD present there if the logbook is not accesible or if that shot is not found on it
- FILDPosition from the DiagParam library was deprecated, to obtain the FILD position, the new FILD class should be used
- load_FILD4_trajectory and plot_FILD4_trajectory where moved to the new FILD library inside LibDataAUG
- load_FILD4_trajectory makes now the conversion between insertion and real R and z. Notice that this is based on CAD and can be non-precise. +- 1 cm can be expected due to failures in the CAD


## 0.7.1 Uncertainties in fits and angles in execution:
### LibMap:
- The fitting routines now return also the uncertainties
- 'Gyroradius_uncertainty' and 'Pitch_uncertainty' were added to the StrikeMap.resolution dictionary
- Strike Map object recognizes which code generated the StrikeMap (thanks to a number in the header which SINPA introduces)
- 'code' and 'version' attributes were added to the Strike Map object
- XI, nXI and uniqueXI attributes were added to the FILD StrikeMap object, as a starting point for we merging of INPA and FILD processing
- The StrikeMap object uses now the new strike points object, common of FILD and SINPA

### LibVideoFiles:
- Included plt_frame_remap() to plot remapped frames

### SINPA
- Added a routine in the SINPA execution library to calculate the FILD orientation following the new criteria
- Recovered the SINPA geometry library which was eliminated by mistake, the calculate rotation matrix is again there
- field object from the common library of the simulation codes now includes a method to generate the field for SINPA given theta and phi, the same 2 angles defined in FILDSIM

### Bug fixed:
- Fixed bug if an old version of Shapely was installed
- Fixed a small bug in the calculation of FILD orientation


## 0.7.0 Common libraries for simulation codes
### Equilibrium
- Included routine to retrieve the flux surface coordinates (R, z).

### i-HIBPsim namelists [iHIBPsim/nml.py]
- Routines to generate generic namelists for the iHIBPsim code [make_namelist]
- Routines to check consistency of namelists [check_namelist].
- Routine to check if the files needed for a run of iHIBPsim are available [check_files]

### i-HIBPsim execution wrapper [iHIBPsim/execute.py]
- prepareRun() wrapper to generate a simple run for iHIBPsim.
- run_ihibpsim() wrapper to run the code properly. No cluster version available.

### i-HIBPsim geometry library [iHIBPsim/geom.py]
- Included particularities of the i-HIBPsim beam model in the library.
- Routines to generate beam lines, divergencies limits...
- gaussian_beam class to handle and contain all the data for a i-HIBPsim beam and plot it.
- geom class contains all the i-HIBPsim geometry: beam, head and scintillator plate and routines to plot it.

### i-HIBPsim beam GUI [GUIs/i-HIBP_beam.py]
- First GUI app for plotting the beam geometry. To be improved with Qt version.
- GUI has to be run by : "run Examples/Others/ihibp_beam_gui.py"

### LibVideo
- Plotting frames and remaps allows for the possibility of using log scale in the colorbar. Just set scale='log'
- Improved efficiency of the counting of saturated frames thanks to build in methods
- flag 'make_copy' from the filter method of the video file was rename to 'flag_copy' to be consistent with the noise subtraction case
- LibVideo split in individual libraries. The complete library was almost 3k lines of code. Now individual libraries are written for each type of archive
- BasicVideoObject created. This object is now the parent class for the INPA, FILD and iHIBP videos. IT just contain the skeleton to read frames, filter them and subtract noise (which is common for all diagnostics). In the future, it will include distortion correction
- FILDVideo object created. Is just the child class of BasicVideoObject with all FILD routines

### LibPlotting
- clean3Daxis() included: It removes the ugly panes that matplotlib puts by default in 3d plots
- axisEqual3D() set aspect ratio to equal in the 3D plot

### Simulation codes
- A new Geometry library was added, it can read geometries from FILDSIM and SINPA code. It can plot in 3D and 2D, shaded and not shared, apply the rotation and translation to the vertex... read the documentation of the library for full detail
- A new StrikePoints object was added. Now is it exactly the same for SINPA and FILDSIM codes!. So from the end user point of veiw, post process the data from both codes is equivalent.  Old FILDSIM strike object left there as for compatibility with all users, but is not recommended

### Deprecation
- The object Geometry from the SINPA library was deprecated. The one from the Common library for the simulations codes should be used!

### Others
- Improved comments and documentation
- The function which read FILDSIM orbits now raise an exception if there were no orbits in the file


## 0.6.5 Interpolators and synthetic signals
- Changed to RBFInterpolator, which seems to be more stable thatn BivariateSpline (#Scipy 1.7.0 or larger is required now#)
- Most robust calculation of the synthetic signal for FILD (no bugs for fcol almost zero)
- Solve a bug in the loading of the strike map. If a StrikePointsFile was passed as argument, the code failed. (Bug introduced in version 0.6.4)

### Deprecations
- p1D() from the plotting library was deprecated


## 0.6.4 New interpolators for SINPA and SMap upgrades
### StrikeMap
- StrikeMap can now be initialize with fild instead of FILD (actually the comparison is lower case, so you can initialize it as FiLd if you are crazy)
- StrikeMap now is able to load strike points from the new FILDSIM format
- If there are not strike points loaded, the function StrikeMap.calculate_resolutions will try to load them
- Plot real updated to show properly the labels if the inputs are in m or cm. Labels are now a bit messy, need a bit more work in future versions
- Smap.sanity_check_resolutions() was deprecated and eliminated
- Smap.plot_resolution_fits() released. This is the new and complete way of plotting the fits performed during the resolution calculation
- Smap.calculate_resolution no longer use predefined indeces but the header object, so it will not be an issue for future changes of strike object files
- _fit_to_model__() now return also de used normalization

### Video
- Video.subtract_noise() was upgrade, loop was eliminated, now is much faster
- Video.subtract_noise() now always return the frame used, the flag return_frame was deprecated

### IO
- IO.save_object_pickle() was corrected. Now it does not fail when user click cancel

### Others
- improved comments and documentations


## 0.6.3 Small improvements
- line_fit_3D was moved from the INPASIM library to the SideFuncitons one
- Change in the SINPA.Strike to accommodate the order changes in SINPA (just a couple of index changed in the header)


## 0.6.2 Small improvements
### TimeTrace
- TimeTrace.plot_single() now shows the axis and include a print for the base line correction done


## 0.6.1
### Mapping library
- plot_resolution allows to plot just the resolution along a given gyroradii, avoiding the 2D contour which is difficult to follow. Check index_gyr new optional variable
- plot_pix of the Scintillator object was upgraded, now 'the scintillator is closed'. Default line style is continuous and color is white

### Video Object
- plot_frame now include by default a colorbar

### SINPA Library
- Solved a bug when the scintillator histogram wanted to be calculated for FILD data

### Enhance plotting
- Lib.Plotting include a function to plot a collection of lines with colors given by a colormap (collection is mapable so you can then include a colorbar)


## 0.6.0 SINPA Support and new Tomography
### Simulation codes
- Libraries to interact with the different simulation codes (FIDASIM, FILDSIM, iHIBPsim, and SINPA) are now located in the SimulationCodes library

### FIDASIM
- Included routines to read the npa data
- Library subdivided in read and plot

### FILDSIM
- a new FILDSIMmarkers library was created. It contain the new object to load and plot the strike maps
- #Note#: This library imply a small change of phylosophy against previous versions. Yuo can still load and use the strike points as before from the strike map, but they are now a part from the FILDSIM library, with their own object and ploting routines.
- This change was made for an earier integration of INPA and for an easier analysis of FILDSIM strike points for FILD optimization
- Function to plot any variable of the FILDSIM strike points was added: see LibFILDSIM.Strikes.plot1D()
- Direct and easy calculation of the histogram of strike points in the scintillator was added: see LibFILDSIM.Strike.calculate_scintillator_histogram() and LibFILDSIM.Strike.plot_scintillator_histogram()
- When the FILDSIM markers are loaded, they are no longer treated like a single matrix, they are splits by pairs (gyroradius, pitch). This save memory (we do not need to save the first 2 colums of the matrix) and simplify routines as the calculation of the resolution
- The function to read the orbits was removed from the FILDSIMexecution library and moved to the FILDSIMmarkers one, inside the new orbtis object
- The same happeded with the plot orbits, which is now a part from the orbit object

### Mapping library
- Support for SINPA strike maps was included in the mapping library
- calculate_transformation_factors was deprecated
- get_points was deprecated
- append_to_database from the database object was deprecated
- The strike points variable of the StrikeMap was completely changed, see the FILDSIM part of the changelog for a full documentation

### SINPA
- the new Synthetic INPA code is supported

### Tomography
- Mono dimensional tomography can be performed, examples can be found in L15

### PC compatibility
- Included a dummy LibData in order to be able to import the suite in your personal PC. Minor modifications here and there in the import statements were done to support this

### Others
- function Lib.LibData.AUG.plot_FILD4_trajectory(shot) renamed to Lib.LibData.AUG.plot_FILD4_trajectory(shot)
- Solved a bug in Video.find_orientation when the function was called with the remap not calculated
- Solve small bug in the plot_real routine of the strike map, before pitch label was 'Pitch [0])' and in the gyroradius one, there were () instead of []
- Vid.plot_orientation no longer set by default the font size, as that is don now when initializing the suite
- Lib.Libfildsim.plot_geometry(). Dummy bug corrected, in the title of the 3 subplot it said 'Y-Z' instead of 'X-Z'
- Default colormap in the GUIS to plot the videos is now grey scale
- Added update_case_insensitive to the Utilities library to compare dictionaries in a case insensitive way
- Added a custom path file so the user can define its own paths
- Improvements in comments + PEP8 checking
- change 'Pablo Oyola:' to 'Pablo Oyola - ' beause Pablo likes more the ' - ' notation to introduce his email
- NBI object includes now an option to plot in 3D
- Solved minor details for the first installation (regarding plotting settings initialization and AUG path)


## 0.5.8 Minor improvements
- The guess_shot of the video class will no longer give an error if the shot number can't be deduced from the file name, it will just return none
- TimeTrace.export_to_ascii() now allows to select the number of digits you want for the output. By default, just 4 digits are used.


## 0.5.7 Minor improvements
- The print netCDF routine of the io is now compatible with netCDF saved without the long _name field
- Upgraded plot_profiles in time, now the labels re-adapt when the user makes zoom


## 0.5.6 i-HIBP namelists and ELM sync routines.
- Added new sublibrary in LibData/AUG names Misc, containing FILD4 trajectories, ELM shotfile...
- Basic namelist generation for i-HIBP simulation codes library included.
- Basic library for i-HIBP beam plotting and marker generator.
- Update in the library BPZ to read and plot BEP fitting data.
- L6 example now uses the MC method
- #Bug solved# related with the single strikemap remap. Before, if you asked the single map remapping, it failed at the end when it tries to save the data, as the variable theta_used was not created, as the theta angle was not evaluated. Now it just save theta_used=0 and solved!


## 0.5.5: Minor improvements and examples
- added an example to plot a discharge overview in AUG
- calculate spectrograms of the fast channel now uses as default the scipy spectrogram function


## 0.5.4: Minor improvements
- plot_single of the TimeTrace object now no longer have default color red, so is not a problem to compare different shots. Line_par and ax_par entries of that functions were renamed to line_params and ax_params to be coherent with the rest of the suite
- new examples to analyse FILD data


## 0.5.3: Minor improvements
- Now the scan of the tomography library saves the data in each interaction (can be deactivate via inputs)
- Label can be set in the plotting of the fast channel via line_params dictionary
- #Bug solved# now the get_fast_signal() will not fail if the requested channel is a component of a numpy array
- Lib.LibData.AUG.plot_FILD4_trayectory(shot) and Lib.LibData.AUG.load_FILD4_trayectory(shot) added to load FILD4 data. First step of FILD4 disclosure
- Plotting style sheet updated, now you can choose default colors for line plotting


## 0.5.2: Minor improvements
- synthetic_signal_remap() will output the signal as a matrix [npitch, nradius] to be consistent with the remap (before it was [nradius, npitch])
- The fast channel options allows now to calculate spectrograms and plot them


## 0.5.1: Fast Channel analysis v1
- synthetic_signal_remap() inputs changed, now gmin, gmax, dg is now renamed as rmin, rmax, dr, to be consistent with the rest of the ScintillatorSuite
- #Bug solved# solves a bug in the synthetic_signal_remap() method, nan where appearing if the markers were outside the map range
- get_fast_channel() from the LibData now also returns the number of the loaded channel


## 0.5.0: New FILD remap
- The 'nearest' method of the interp_grid was deprecated
- The interp_grid method of the StrikeMap class was completely rewritten, please see the new function
- The remap method will call interp_grid of the smap object instead of failing if the grid was not interpolated before calling this function
- inputs for remap method of the mapping library was changed, now the edges of the histogram should be calculated outside (improve efficiency and easily allows for MC or standard remap switch)
- New MC remap based in the 'Translation Tensor' developed. See documentation PDF for a full description of the method


## 0.4.15 Profile routines and EHO tracker.
- Toroidal rotation reading routines has been included: from PED, IDI or make a smoothing spline to the CXRS raw data.
- Routines to read the profiles (electron temperature and density) from PED.
- EHO tracker with and without diamagnetic corrections has been included in Examples.
- Phase correction for the magnetic pick-up coils in AUG is now included.
- The phase correction files are automatically downloaded at the first time that the magnetic routines from AUG are run.


## 0.4.14: Smap and plotting improvements
- The StrikeMap object can now be initialised with the theta and phi angle, no longer need the full path to the file (although of course you can still use the file)
- If no file is given to the StrikeMap.load_strike_points() the code will look for the strike points file in the same folder than the strike map
- The substract noise function include now an option to make a copy of the frames or not (to save memory, dafult: True)
- Default plotting options now available via configurable namelist (Data/MyData)
- Minor ToDos solved
- Upgraded Readme


## 0.4.13: FILDSIM forward modeling
- Camera parameters no longer in LibParams but in separate txt files in the Data folders
- f90mnl is now a fundamental module, the suite will not work without it
- Added function in the LibIO to read the camera properties
- Current synthetic_signal and plot_synthetic signal function of the FILDSIM library renamed to synthetic_remap and plot_synthetic_remap
- #Note#: The weight function calculation does no longer include  # dr_scint # dp_scint, so the W has dimension of one over dgyr and dpitch of the scintillator grid used for the calculus
- Several plotting plotting capabilities added (credit to Ajvv)
- Routines to model basic camera noise added


## 0.4.12: Small improvements
- New examples for the tracker were added
- #Note#: The order of the inputs in the function write_markers for the tracker was changed, to follow the same logical order of the rest of the suite, now is: write_markers(markers: dict, filename: str)
- Small PEP8 stile corrections
- functions to save and read objects with pickles were added, this allows to save and load figures more or less as .fig from matlab (see save_object_pickle and load_object_pickle)
- Update run_paths.py to the new system to import modules
- function to read the deposition markers was added
- old method to write tracker namelist recoverd for legacy compatibility
- #Note#: the input of the LibIHIBPorbits, for the plot, is now 'ax_params' and 'line_params' instead of 'ax_options' and 'line_options', to be consistent with the rest of the suite
- #Note#: the input of the LibIHIBfields, to read the magnetic field from the database, now requiers shot and time instead of time and shot, to be consistend with the rest of the suite
- #Note#: same with readPsiPolfromDB
- #Note#: vt renamed to vphi in the properties of the markers


## 0.4.11: HotFix
- Fix an issue while importing library of BEB
- change '()' on the plot strike map for '[]' (all the rest of the plots of the suite indicate the units between [])


## 0.4.10: Tomography improvements
### Tomography improvements
- Solved a bug in the process to W2D to W4D, last gyroradii was being ignored
- Now fildsim.build_weight_matrix() gives also the W2D matrix
- Lib.Tomography.prepare_X_y_FILD now can apply a median filter to the remap frame
- Forward modeled frame and profiles included in the Tomography GUI

### NBI improvements
- Renamed _NBI_diaggeom_cordinates to NBI_diaggeom_cordinates
- The function NBI_diaggeom_coordinates include now the 'length' of the NBI line as well as the tangency point
- Included 'calculate_intersection' method in the NBI class to calculate the intersection points of the NBI line with the flux surfaces
- Included generate_tarcker_markers in the NBI class to generate markers for the tracer

### Tracker changes
- The write namelist for the tracker was updated to the new f90mnl format adapted in the rest of the suite
- Duplicated tracker routines were eliminated, now only the iHIBPsim library should be used for the fields and orbits reading
- #DEPRECATED# The flag grid on the plotTimeTraces() of the orbit class was deprecated, if you want to plot the grid pass grid:'both' or 'major' to the ax_options dictionary
- plotTimeTraces() now has a flag to plot the R,Z,phi temporal evolution
- The routines to plot the orbits now admit a flag (default: True) to plot the vessel or not
- Added routine in the orbit class to calculate the gyrocenter coordinates
- Added the possibility of calculating the magnetic moment with the gyrocenter Bfield

### Forward modeling improvements:
- Include check to avoid the forward modeling routine to give Nan when some points of the distribution are outside the range of the Strike map, these points will be ignored


## 0.4.8: Toroidal rotation fitting and hotfix for magnetic spectograms:
### LibData
- Introduction of routines to read the toroidal rotation velocity from AUG database. Available profiles from IDI, PED and spline-regression to several CXRS diagnostics (CUZ, COZ, CMZ & CEZ).

### Magnetics
- Ballooning coils phase correction for the FFT taken from pyspecview.
- All examples in FreqAnalysis corrected with the phase.


## 0.4.7: Support for BEP plotting:
- Added initial library for reading the calibrated and uncalibrated signal from BEP shotfiles.
- Simple GUI to plot interactively see the spectra for shots.
- Added few examples to plot the BEP in a non-interactive way.


## 0.4.6: FILDSIM orbit plotting:
- Orbit plotting included to plot FILDSIM calculated orbits


## 0.4.5: Bug solved:
- Solved a bug in the diaggeom coordinates for NBI8. NBI8 end was off by almost 10 cm


## 0.4.4: Import changes:
- Routes to libraries were change such that you can import the library just setting your environment variable in the path


## 0.4.3: i-HIBPsim strike line reader & Frequency tracking.
### LibHIBPstrikes
- Adding support read and plot the strikelines from i-HIBPsim code.
- Added support to plot the scintillator synthetic signal.
- Added support to introduce the database of strike lines.
- Changed attributes in the database to adapt to a common TRANSP-like database. long_name contains a full description of the field while the short_name contains a ready-to-plot name.

### LibFrequencyAnalysis
- Added STFT2 routine: wrapper to scipy implementation, emulating Giovanni's.
- Added iSTFT routine: wrapper to scipy implmentation, to reconstruct the signal from an STFT.
- Added Vertex and Graph classes, allowing for minimal path search (using Dijsktra's method).
- Added routine to search for frequency in a spectrogram (trackFrequency).
- Moved examples 'multipow', 'frequencyTracking' to new Folder: 'FrequencyAnalysis'
- New example to plot fast the spectrogram of a given magnetic pick-up coil.

### Movement of LibDataAUG
- LibDataAUG is now moved inside the folder LibData, to allow for a smother integration of future machines


## 0.4.2: FILDSIM forward modeling
- Now the StrikeMap.calculate_resolutions() also calculate the interpolators so one can just call smap.interpolators['pitch']['sigma'](gyr0, pitch0) and you will have the interpolated value of sigma of the pitch for gyr0, pitch0.
- The StrikeMap object for FILD now include the fields: unique_gyroradius, unique_pitch and collimator_factor_matrix.
- #Included requested feature#: Issue #58: read_ASCOT_distribution implemented, only valid for ASCOT4
- Fits of the calculate resolution function are now inside the 'fits' dictionary, contained in the resolution section of the strike map object
- read_scintillator_efficiency moved from the LibIO to the new LibScintillatorCharacterization.py
- Efficiency included in FILD forward modeling
- Efficiency included in tomography
- Calculation of the W function for FILD re-written in a more compact way. Coherent with the models used to calculate the resolutions. Now it much faster
- fildsim.plot_geometry added in the fildsim library. It plot the plates geometry in 3d and is projections
- Method relating the absolute calibration of the frames removed from FILDSIM library, they'll be included again in next version once they are tested


## 0.4.1: Minor improvements + ELM filtering
- Added a function to calculate the intersection between any curves in 2D (LibUtilities.find_2D_intersection(x1, y1, x2, y2))
- Improved LibPlotting.plot_flux surfaces() : Now color can be selected, cm can be used as units, the axis limit will not be changed if an axis is given
- Included root directory of the suite in path_suite.py to be aable of using the command =import Lib as ss= outside the root directory of the suite
- Included reading of ELM time base (LibData.profiles.get_ELM_timebase.py)
- Included ELM filtering: Note, it will just delete from your input signal the ELM time points
- Read frame from a cin file will no longer return a squeeze matrix when you load the frames internally. When you load them externally, they will be squeeze()
- Solved issue #7: NBI profile calculation and plot upgraded
- Plot NBI added to the NBI class
- Now calc_pitch_profile of the NBI class take as default IpBt sign defined in the .dat library


## 0.4.0: New suite structure:
- Typos in comments corrected
- PEP8 agreement revised
- LibDataAUG subdivided in different modules (it was too big)
- Re-written first_run.py
- Verbose of remap_all_loaded_frames_FILD.py improved


## 0.3.6: Improvements in tomography:
- Now the Ridge, nnridge and Elastic net scan also return a dictionary with the produced figures


## 0.3.5: Bug solved:
- #Bug solved# Solved issue #54 on the broken time base of CCD cameras


## 0.3.4: First INPASIM utilities:
- GUIs files where divided into a new folder GUIs
- #Included requested feature#: Issue #33. Now if a path is passed to the remap routine mask=path the code will load the mask contained in file inidcated by path
- Included Non Negative Ridge as a regression method
- Included method to cut the video in the Video class, to restrict to a given region of pixels: Video.cut_frames()
- A flag was added in the noise_subtraction and filter methods of the Video class in order to decide if we want to create a copy of the experimental frames or not
- First methods to calculate optical transmission


## 0.3.3 i-HIBPsim strikeline and strikes reader:
- New library under iHIBPsim for reading and plotting strikelines and strikes on the scintillator.
- Added function in LibDataAUG for reading magnetic pick-up coils and group of them (same toroidal location).
- Added function in LibDataAUG for reading from the equilibrium the basics of the shot data (Bt0, Ip, elongation, ...)


## 0.3.2: First INPASIM utilities:
- Added function to fit a line to a 3d cloud of points
- Rewritten paths_suite.py to allow make easier to include new libraries


## 0.3.1: Tomography:
- Update examples to the new version
- Updated Smaps library (more maps) download the new version if you want
- video.find_orientation() added, allows to find the calculated theta and phi (Yes, I was lazy and I've created a small function to avoid the calculation of this manually)
- Now the same criteria of rmin, dr and so on is implemented in the tomographic reconstruction section
- Scan of tomographic reconstruction now gives a dict as output, not single outputs
- New GUI for tomographic representation plotted


## 0.3.0: GUIs and plotting
- Simplified StrikeMap.plot_pix() and StrikeMap.plot_real(). #IMPORTANT# Names of the input arguments were changed!!!
- Included GUI to explore the camera frames, Video.plot plot_frames_slider() was rename as Vide.GUI_frames()
- Included GUI to explore the remapped frames, Vide.GUI_frames_and_remap()
- Improved Video.plot_frame() was upgraded now you can write 'auto' and the function will load and plot the StrikeMap (see its documentation for further instructions)
- LibPlotting.remove_lines() added, it deletes all lines from a plot, useful to delete the strikemap of one of your plots (used by the new GUIs)
- #BUG SOLVED#: Selecting 'cancel' in the export remap windows raised and error. Now it solved


## 0.2.9 Multipow calculation.
- Included functions to read magnetic coils in LibDataAUG
- Included functions to read the ECE data in LibDataAUG.
- Included plotting function for the ECE data in LibPlotting
- Included plotting function for flux surfaces using contour levels.
- Solved hotfix for the 0.2.8
- Multipow (CPSD for magnetics-ECE) included as an Example/Others
- Included myCPSD calculation for cross-power calculation in LibFrequencyAnalysis.


## 0.2.8 i-HIBP cross sections.
- Included i-HIBP cross sections calculation and storing to files (Issue 34)


## 0.2.7: Hot fix
- #BUG_SOLVED# Problem with the name of the number of saturated pixels solved, now it is possible to export the remap again (the bug was introduced in version 0.2.6). Issue #50


## 0.2.6: Count pixels
- #Included requested feature#: Issue #50 now the number of pixels over a given threshold is counted by default. User can set this threshold in the read_frame method of the video object
- Video.plot_number_saturated_counts() added. If executed without arguments, it plot the pixels counted by default when reading the video. The function accept also a threshold, in this case the pixels are count again
- #BUG_SOLVED# The angles of rFILD are now properly included


## 0.2.5: Improvements in the remap
- Now when some Smap is missing, the program will give the option to use the nearest (in time) existing strike map
- The real value of theta (with all the decimals) as well as the used one are stored to compare the angles used in the remap
- Added plot_orientation() to the video object, to plot the calculated angles with the orientation (real and used)
- The method fildsim.write_namelist() now overwrite by default the existing namelist. You can change this behavior with the flag =overwrite=
- The method fildsim.guess_strike_map_name_FILD() now do not create extra strike maps like 0 and -0
- Camera model included as one more data in the FILD dictionary in LibDataAUG.py
- Some PEP8 correction in iHIBP library


## 0.2.4: HotFix
- #BUG_SOLVED# Solved bugs in the LibFILDSIM.find_strike_map routine, the fildsim options were not updated properly
- Updated FILDSIM example following new f90nml requirements


## 0.2.3: Filter for video object
- #Included requested feature#: median filter added to the filter_frames method of the video class (closes #47)
- #Closes #45# Now the rmin, rmax, pmin, pmax represent the output vector when we want the remap, not the input edges (:-()
- 'Clean' a bit the method 'find_strike_map' from the FILDSIM library, now a loop is used to run over FILDSIM namelist
- Included Gaussian filter for the video frames
- Reordered examples
- Simplified plotting options in TimeTrace.plot_single()
- Improved TimeTrace.plot_all(), now they share x axis so zoom is better


## 0.2.2: Debugging
- #BUG_SOLVED# in the plot_vessel function, the factor from m to cm was 10 instead of 100!
- #BUG_SOLVED# rotation of the vessel was not passed from the plot_vessel routine to the method which calculate the vessel coordinates
- #BUG_SOLVED# Solved bug when the requested interval to average the noise was not in the file (issue #46)


## 0.2.1: FIDASIM implementation
- First routines to read FIDASIM output added, (thanks Pilar :-)) Although some work still needed in that module this is not completely checked
- Updated Readme following nice example of iHIBP
- Calibration used in the remapping is saved in the remapping options, such that future comparisons of remapped data is easier
- plot_profiles_in_time of the video object allows now to pass the min and max of the scale as inputs
- #Included requested feature# First implementation of issue #41


## 0.2.0: Strike Maps reordering
- p1D_shaded_error updated with the possibility of plotting the central line
- Updated gitignore to ignore a folder call 'MyRoutines' for the user to have its own routines
- Updated the paths to strike maps, now two libraries will be used: Remap 'low' number of markers, 'Tomography' high number of markers
- Updated namelist format, now the suite follows the criteria given in the f90nml module
- Added GNU license


## 0.1.9: Spectrograms
- First spectrogram function added, first step towards the fast channel analysis
- Better examples included
- Better checking of whether we are in AUG or not
- Now the remapping of the whole shot can be done using a given strike map


## 0.1.8: Reverse FILD
- IB sign were included to include the proper pitch definition in FILDSIM even with the reverse field
- #BUG_SOLVED#: Solve a bug which forced the remap to ignore theta and phi if just one of the strike maps was not found
- Included the RealBPP in the exported remap data


## 0.1.7: Improve reading/writing
- Solved the issue in the init due to new iHIBPsim libraries
- Included a check to not overwrite files, now if one of the saving routines try to save a file which exist, it will open a window to give to the user the chance to change the name
- Added also a similar function to open files in case it does not find the name, it will pop-up a window
- Improved the checking to test we are in AUG
- Add a method to integrate the remapped frames in the desired range radius-pitch (arbitrary shapes allowed via roi)


## 0.1.6: What's new?
- Added possibility of loading the used ROIs
- Added the possibility of plotting each individual time trace
- Added general routine to load the created ncdf files
- Suppressed remapped slider plotting in the video object, it was too buggy, new one will come with tkinter


## 0.1.5: What's new?
- Now the remap_all_loaded_frames_FILD first calculate all theta and phi and see how many strike maps must be calculated. The user can decide whether if perform the FILDSIM calculation or just take a single strike map
- Added the possibility of remapping with a ROI. Also export the ROI


## 0.1.4: What is new?
- iHIBP routines to interact with the tracker and iHIBPsim, first round
