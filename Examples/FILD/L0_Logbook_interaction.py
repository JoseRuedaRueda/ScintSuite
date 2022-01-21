"""
Load video from FILD cameras

Lesson 0 from the FILD experimental analysis. How to interact with the FILD
logbook

jose Rueda: jrrueda@us.es

Note; Written for version 0.7.8

You should run paths_suite.py before runing this example.
"""
import Lib as ss
from pprint import pprint

# --- Initialise the logbook
logbook = ss.dat.FILD_logbook()  # you can define your custom paths to the
#                                # excel containing the positions and the
#                                # camera txt etc, but the default should be ok
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
cal = logbook.getCameraCalibration(shot=39612, camera='PHANTOM', diag_ID=1)
# unfortunatelly the camera input is neecesary, maybe I create another database
# to handle this information, but is a chaos for old shots...
print('--- Camera calibration')
cal.print()  # call is a custom class, it has in own print

# - Which are all shots where rFILD collimator was used? (FILD inserted)
# rFILD is labeld as id1 because is the first collimator and head geometry
# which was exaustivelly documented
print('--- Making first search')
shots_Rfild = logbook.getGeomShots('id1')
print('%i shots found' % shots_Rfild.size)
# - What is I only want shots in which FILD is inserted beyond 2.19m?
print('--- Making second search')
shots_RFILD_deep_insertion = logbook.getGeomShots('id1', maxR=2.19)
print('%i shots found' % shots_RFILD_deep_insertion.size)
