import ScintSuite as ss
from pprint import pprint
import numpy as np
import ScintSuite.LibData as ssdat
from rich import inspect


geomID = 'MU01'
shot = 54277

## debug __init__ of _FILDVideoObject
diag_ID = 1
logbookOptions = {}

file = ssdat.guessFILDfilename(shot, diag_ID) # W

## this seems to be the issue. on line 134, no other values are passed to the logbook
logbook = ssdat.FILD_logbook(**logbookOptions) 


# ## open logbook
# logbook = ss.dat.FILD_logbook(positionFile="/home/fn2394/MAST-U_FILD_logbook_v3.xlsx")

# ## now debug "getPosition" function
# FILDid = 1

# ###
# geomID = logbook.getGeomID(shot, FILDid)
# default = logbook._getPositionDefault(geomID)

# position = {        
#             'R': 0.0,
#             'z': 0.0,
#             'phi': 0.0,
#         }        

# if not logbook.flagPositionDatabase:
#             print("oops")
# else:
#     # Get the shot index in the database
#     if shot in logbook.positionDatabase['shot'].values:
#         i, = np.where(logbook.positionDatabase['shot'].values == shot)[0]
#         flag = True # True when the shot is in the database
#     else:
#         # logger.warning('Shot not found in logbook')
#         print("oops")

# dummy = logbook.positionDatabase['FILD'+str(FILDid)]
#     if shot in logbook.positionDatabase['shot'].values:
#         i, = np.where(logbook.positionDatabase['shot'].values == shot)[0]
#         flag = True # True when the shot is in the database
#     else:
#         # logger.warning('Shot not found in logbook')
#         print("oops")

# dummy = logbook.positionDatabase['FILD'+str(FILDid)]