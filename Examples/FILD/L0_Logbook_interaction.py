"""
Load video from FILD cameras

Lesson 0 from the FILD experimental analysis. How to interact with the FILD
logbook

jose Rueda: jrrueda@us.es

Note; Written for version 0.7.8. Check for version 0.8.0

You should run paths_suite.py before runing this example.

Notice that this example is AUG oriented. In all supported machines the logbook
has the same public methods, which works in the same way (same inputs/outputs)
but each machine can have a particular set of files and options to initialise
Therefore the example lines around lines 30 can be different in your machine
Please have a look at the help of your logbook object
"""
import Lib as ss
from pprint import pprint

# --- Initialise the logbook
logbook = ss.dat.FILD_logbook()  # you can define your custom paths to the
#                                # excel containing the positions and the
#                                # camera txt etc.
# For example, if I want to load the camera calibration data base, from my own
# file, 'MyCalib.txt':
# logbook = ss.dat.FILD_logbook(cameraFile='MyCalib.txt')
# if you want to load your particular excel file with the position
# logbook = ss.dat.FILD_logbook(positionFile='MyNiceExcel.xls')

# --- get stuff
# - Which head configuration was installed in FILD1 manipulator in #39612?
geomID = logbook.getGeomID(shot=39612, FILDid=1)
print('The geometry was: ', geomID)
# - Where was this FILD?
position = logbook.getPosition(shot=39612, FILDid=1)
pprint(position)
# - Which was its orientation?
orientation = logbook.getOrientation(shot=39612, FILDid=1)
# - Which was the camera calibration from that shot?
cal = logbook.getCameraCalibration(shot=39612, diag_ID=1)
# Notice that this is coming from old core of the Suite, where diag_ID was used
# we mantained here for convenience for the old users, although the routine
# accept also FILDid as input, this is just an alias for diag_ID
print('--- Camera calibration')
cal.print()  # call is a custom class, it has in own print

# - Which are all shots where RFILD collimator was used? (FILD inserted)
# rFILD is labeld as AUG01 because is the first collimator and head geometry
# which was exaustivelly documented
print('--- Making first search')
shots_Rfild = logbook.getGeomShots('AUG01')
print('%i shots found' % shots_Rfild.size)
# - What if I only want shots in which RFILD is inserted beyond 2.19m?
print('--- Making second search')
shots_RFILD_deep_insertion = logbook.getGeomShots('AUG01', maxR=2.19)
print('%i shots found' % shots_RFILD_deep_insertion.size)
