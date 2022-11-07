"""
Load video from FILD cameras

Lesson 0 from the FILD experimental analysis. How to interact with the FILD
logbook

jose Rueda: jrrueda@us.es
Lina Velarde: lvelarde@us.es

Note; Written for version 0.7.8. Check for version 1.0.0

Notice that this example has only been tested for MU. In all supported machines
the logbook has the same public methods, which works in the same way (same
inputs/outputs) but each machine can have a particular set of files and options
to initialise. Therefore the example lines around lines 21 can be different in
your machine.
Please have a look at the help of your logbook object
"""
import Lib as ss
from pprint import pprint

# -----------------------------------------
# --- Inpus
# -----------------------------------------
geomID = 'MU01'
shot = 44732

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
# - Which head configuration was installed in FILD1 manipulator in #shot?
geomID = logbook.getGeomID(shot=shot, FILDid=1)
print('The geometry was: ', geomID)
# - Where was this FILD?
position = logbook.getPosition(shot=shot, FILDid=1)
print(position)
# - Which was its orientation?
orientation = logbook.getOrientation(shot=shot, FILDid=1)
# - Which was the camera calibration from that shot?
cal = logbook.getCameraCalibration(shot=shot, diag_ID=1)
# Notice that this is coming from old core of the Suite, where diag_ID was used
# we mantained here for convenience for the old users, although the routine
# accept also FILDid as input, this is just an alias for diag_ID
print('--- Camera calibration')
cal.print()  # call is a custom class, it has in own print

# - Which are all shots where RFILD collimator was used? (FILD inserted)
# rFILD is labeld as AUG01 because is the first collimator and head geometry
# which was exaustivelly documented
print('--- Making first search')
shots_Rfild = logbook.getGeomShots(geomID)
print('%i shots found' % shots_Rfild.size)
# - What if I only want shots in which RFILD is inserted beyond 2.19m?
print('--- Making second search')
shots_RFILD_deep_insertion = logbook.getGeomShots(geomID, maxR=2.19)
print('%i shots found' % shots_RFILD_deep_insertion.size)

# - Did we have overheating in FILD1 during a given shot?
# - (Only for AUG)
if geomID != 'MU01':
    overheating = logbook.getOverheating(shot, FILDid=1)
    print('--- Checking overheating:')
    print('Overheating level for shot 41255: %i' % overheating)
