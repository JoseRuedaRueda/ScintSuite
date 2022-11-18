"""
NBI geometry package provided by Niek den Harder with the actual NBI geometries.

Niek den Harder - niek.den.harder@ipp.mpg.de

Adapted and numb-ified to the ScintSuite by Pablo Oyola - poyola@us.es
"""

import numpy as np
import numba as nb

try:
    from yaml import CLoader as yaml_read
except ImportError:
    from yaml import Loader as yaml_read

#---------------------------------------------------------------------
#----------------------- Geometry definition -------------------------
#---------------------------------------------------------------------

@nb.njit(nogil=True)
def make_beamlets_box1_uwvcords_steer_angle(gridcord: float,
                                            griddir1: float,
                                            griddir2: float,
                                            sourangl: float=None):
    """
    This function generates the uwv coordinates of the beamlets, and the 2
    normal directions for box 1. Also incorporates steering of the sources.
    I chose to all code this into 1 big routine so that all the small
    parts can be checked.

    Niek den Harder
    """
    if sourangl is None:
        sourangl = np.zeros((4, 2))

    assert sourangl.shape[0] == 4, 'The input angles do not broadcast!'
    assert sourangl.shape[1] == 2, 'The input angles do not broadcast!'

    # Will contain the offset of the set screws that steer the sources.
    # Is calculated from then angles given in sourangl.

    adjudisp = np.zeros((4,2))
    for i in range(len(sourangl)):
        # The numbers are the distances between the ball joint and the
        # adjusting screws.
        adjudisp[i,0] = -335.0E-3*np.tan(sourangl[i,0]*np.pi/180.0)

        # A more positive angle should result in a negative offset for
        # both sources.
        adjudisp[i,1] = -897.0E-3*np.tan(sourangl[i,1]*np.pi/180.0)

    # These values are taken from the DB_B03_BL2 Datenblatt which gives
    # the beam geometry for injector 1.
    qlocations = np.array([[6500., 470., 600.],
                           [6500., -470., 600.],
                           [6500., -470., -600.],
                           [6500., 470., -600.]])*1.0E-3
    qlocation2 = np.array([[-500., -36.2, 0.0],
                           [-500., 36.2, 0.0],
                           [-500., 36.2, 0.0],
                           [-500., -36.2, 0.0]])*1.0E-3

    qnormalsv = qlocation2-qlocations
    Qtransfor = np.ones((len(qnormalsv),3,3))

    # This borrows heavily from https://en.wikipedia.org/wiki/Transformation_matrix
    # The idea is that the basis vectors (x,y,z) should transform to (u,v,w)
    for i in range(0, len(qnormalsv)):
        qnormalsv[i] /= np.linalg.norm(qnormalsv[i]) # Normalize

        #This is to swap the y and z coordinate and reverse the x axis.
        Qtransfor[i] *= np.array([[-1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0]]).T

    # The turning point is determined in a discussion with Rudi Riedl
    # at 2016-07-12, ~9:30, and 2017-01-17 ~15:00 to determine that for
    # box 1 it is identical to the non-CD sources in box 2.
    # Based on DB_F01_BL1 the signs are determined. #
    # In short, the turning points are all in the middle,
    # so below source 1 and 2, and above source 3 and 4.
    turnpoin = np.array([[-(756.6+183.-14.0), 0.0, -430.0],          #Source 1 Location of ball joint in reference to the origin at the beam axis at the location of the end of the earth grid.
                         [-(756.6+183.-14.0), 0.0, -430.0],          #Source 2 u is the beam axis (with the beam towards negative u)
                         [-(756.6+183.-14.0), 0.0, +430.0],          #Source 3 v is the horizontal direction (to the right in the beam direction is positive)
                         [-(756.6+183.-14.0), 0.0, +430.0]])*1.0E-3  #Source 4 w is the vertical direction (up is positive)
    adj1poin = np.array([[-(756.6+183.-14.0), +335.0, -430.0],          #Source 1 Location of the first turning point
                         [-(756.6+183.-14.0), -335.0, -430.0],          #Source 2
                         [-(756.6+183.-14.0), -335.0, +430.0],          #Source 3
                         [-(756.6+183.-14.0), +335.0, +430.0]])*1.0E-3  #Source 4 and conversion from mm to m
    adj2poin = np.array([[-(756.6+183.-14.0), 0.0, +(-430.0+897.0)],         #Source 1 Location of the second turning point
                         [-(756.6+183.-14.0), 0.0, +(-430.0+897.0)],         #Source 2
                         [-(756.6+183.-14.0), 0.0, -(-430.0+897.0)],         #Source 3
                         [-(756.6+183.-14.0), 0.0, -(-430.0+897.0)]])*1.0E-3 #Source 4 and conversion from mm to m

    adj1disp = np.zeros((4, 3))
    adj2disp = np.zeros((4, 3))
    for ii in range(adj1poin.shape[0]):
        adj1disp[ii, 0] = adjudisp[ii, 0]
        adj2disp[ii, 0] = adjudisp[ii, 1]

    nosourcs = len(qnormalsv)
    nobeamls = len(gridcord)
    trancord = np.zeros((nosourcs*nobeamls,3))
    trandir1 = np.zeros((nosourcs*nobeamls,3))
    trandir2 = np.zeros((nosourcs*nobeamls,3))
    for i in range(0, nosourcs): #Loop over the sources
        # These are the angles due to the tilting of the sources inside the box.
        normyang = -np.arcsin(qnormalsv[i,2])
        normzang = np.arcsin(qnormalsv[i,1]/np.cos(normyang))

        # The first adjustment point is horizontal
        # This is for the determination of the tilting due to beam steering.
        adj1vnor = adj1poin[i] - turnpoin[i] + adj1disp[i]

        # The second adjustment point is vertical
        adj2vnor = adj2poin[i] - turnpoin[i] + adj2disp[i]

        #This is the normal vector of the source in the adjusted position
        adj3vnor = np.cross(adj1vnor, adj2vnor)
        adj3vnor /= np.linalg.norm(adj3vnor)

        # This ensures that all the sources point towards negative u.
        # Note that the beam steering does not prevent that, so in all
        # realistic cases this is correct.
        adj3vnor *= -np.sign(adj3vnor[0])

        # These are the angles due to the additional beam steering.
        steryang = -np.arcsin(adj3vnor[2])
        sterzang = np.arcsin(adj3vnor[1]/np.cos(steryang))

        # Procedure: translated, tilted due to steering, backtranslated,
        # tilted due to box angle, put into position.
        for j in range(0, nobeamls):
            tempcord = np.dot(Qtransfor[i], gridcord[j]) - turnpoin[i]

            # Then the source coordinates are tilted (due to steering).
            tempcord = modi_rotate_3D_point(tempcord,
                                            np.array([0.0, -steryang, -sterzang]))

            # The normal vectors are tilted (due to steering)
            tempdir1 = modi_rotate_3D_point(np.dot(Qtransfor[i], griddir1[j]),
                                            np.array([0.0, -steryang, -sterzang]))
            tempdir2 = modi_rotate_3D_point(np.dot(Qtransfor[i], griddir2[j]),
                                            np.array([0.0, -steryang, -sterzang]))

            # Then the source coordinates are tilted (due to box geometry).
            trandir1[i*nobeamls + j] = modi_rotate_3D_point(tempdir1,
                                                            np.array([0.0, -normyang, -normzang]))
            trandir2[i*nobeamls + j] = modi_rotate_3D_point(tempdir2,
                                                            np.array([0.0, -normyang, -normzang]))

            # And the source coordinates are tilted and back translated.
            trancord[i*nobeamls + j] = modi_rotate_3D_point(tempcord + turnpoin[i],
                                                            np.array([0.0, -normyang, -normzang])) + qlocations[i]

    return trancord, trandir1, trandir2



@nb.njit(nogil=True)
def make_beamlets_box2_uwvcords_steer_angle(gridcord: float,
                                            griddir1: float,
                                            griddir2: float,
                                            sourangl: float=None):
    """
    This function generates the uwv coordinates of the beamlets, and the 2
    normal directions for box 2. Also incorporates steering of the sources.
    I chose to all code this into 1 big routine so that all the small
    parts can be checked.

    Niek den Harder


    """
    if sourangl is None:
        sourangl = np.zeros((4, 2))

    assert sourangl.shape[0] == 4, 'The input angles do not broadcast!'
    assert sourangl.shape[1] == 2, 'The input angles do not broadcast!'

    adjudisp = np.zeros((4,2))
    for i in range(len(sourangl)):
        # The numbers are the distances between the ball joint and the
        # adjusting screws.
        adjudisp[i,0] = -335.0E-3*np.tan(sourangl[i,0]*np.pi/180.0)

        # A more positive angle should result in a negative offset
        # for both sources.
        adjudisp[i,1] = -897.0E-3*np.tan(sourangl[i,1]*np.pi/180.0)

    # These values are taken from the DB_B09_BL2 Datenblatt which gives the
    # beam geometry for injector 2.
    qlocations = np.array([[6500., 470., 600.],
                           [6500., -470., 700.],
                           [6500., -470., -700.],
                           [6500., 470., -600.]])*1.0E-3
    qlocation2 = np.array([[-500., -36.2, 0.0],
                           [ 500., -36.2, 0.0],
                           [500., -36.2, 0.0],
                           [-500., -36.2, 0.0]])*1.0E-3
    qnormalsv = qlocation2-qlocations
    Qtransfor = np.ones((len(qnormalsv),3,3))

    # This borrows heavily from https://en.wikipedia.org/wiki/Transformation_matrix
    # The idea is that the basis vectors (x,y,z) should transform to (u,v,w)
    for i in range(len(qnormalsv)):
        qnormalsv[i] /= np.linalg.norm(qnormalsv[i]) # Normalize

        # This is to swap the y and z coordinate and reverse the x axis.
        Qtransfor[i] *= np.array([[-1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0]]).T

    # The turning point is determined in a discussion with Rudi Riedl at
    # 2016-07-12, ~9:30, and 2016-07-13, ~9:30 (added 35 mm for the source
    # flange thickness). This only applies to the CD sources.
    # Based on DB_F01_BL1 the signs are determined.
    # In short, the turning points are all in the middle,
    # so below source 1 and 2, and above source 3 and 4.
    turnpoin = np.array([[-(756.6+183.-14.0), 0.0, -430.0],          #Source 5 Location of ball joint in reference to the origin at the beam axis at the location of the end of the earth grid.
                         [-(756.6+35.0+42.5), 0.0, -430.0],          #Source 6 u is the beam axis (with the beam towards negative u)
                         [-(756.6+35.0+42.5), 0.0, +430.0],          #Source 7 v is the horizontal direction (to the right in the beam direction is positive)
                         [-(756.6+183.-14.0), 0.0, +430.0]])*1.0E-3  #Source 8 w is the vertical direction (up is positive)
    adj1poin = np.array([[-(756.6+183.-14.0), +335.0, -430.0],          #Source 5 Location of the first turning point
                         [-(756.6+35.0+42.5), -335.0, -430.0],          #Source 6
                         [-(756.6+35.0+42.5), -335.0, +430.0],          #Source 7
                         [-(756.6+183.-14.0), +335.0, +430.0]])*1.0E-3  #Source 8 and conversion from mm to m
    adj2poin = np.array([[-(756.6+183.-14.0), 0.0, +(-430.0+897.0)],         #Source 5 Location of the second turning point
                         [-(756.6+35.0+42.5), 0.0, +(-430.0+897.0)],         #Source 6
                         [-(756.6+35.0+42.5), 0.0, -(-430.0+897.0)],         #Source 7
                         [-(756.6+183.-14.0), 0.0, -(-430.0+897.0)]])*1.0E-3 #Source 8 and conversion from mm to m

    adj1disp = np.zeros((4, 3))
    adj2disp = np.zeros((4, 3))
    for ii in range(adj1poin.shape[0]):
        adj1disp[ii, 0] = adjudisp[ii, 0]
        adj2disp[ii, 0] = adjudisp[ii, 1]

    nosourcs = len(qnormalsv)
    nobeamls = len(gridcord)
    trancord = np.zeros((nosourcs*nobeamls,3))
    trandir1 = np.zeros((nosourcs*nobeamls,3))
    trandir2 = np.zeros((nosourcs*nobeamls,3))
    for i in range(nosourcs): #Loop over the sources

         #These are the angles due to the tilting of the sources inside the box.
        normyang = -np.arcsin(qnormalsv[i,2])
        normzang = np.arcsin(qnormalsv[i,1]/np.cos(normyang))

        # The first adjustment point is horizontal
        # This is for the determination of the tilting due to beam steering.
        adj1vnor = adj1poin[i] - turnpoin[i] + adj1disp[i]

        # The second adjustment point is vertical
        adj2vnor = adj2poin[i] - turnpoin[i] + adj2disp[i]

        # This is the normal vector of the source in the adjusted position
        adj3vnor = np.cross(adj1vnor, adj2vnor)
        adj3vnor /= np.linalg.norm(adj3vnor)

        # Sources point towards negative u.
        # Note that the beam steering does not prevent that, so in all
        # realistic cases this is correct.
        adj3vnor *= -np.sign(adj3vnor[0])

        # These are the angles due to the additional beam steering.
        steryang = -np.arcsin(adj3vnor[2])
        sterzang = np.arcsin(adj3vnor[1]/np.cos(steryang))

        # Procedure: translated, tilted due to steering, backtranslated,
        # tilted due to box angle, put into position.
        for j in range(nobeamls):
            tempcord = np.dot(Qtransfor[i], gridcord[j]) - turnpoin[i]

            # Then the source coordinates are tilted (due to steering).
            tempcord = modi_rotate_3D_point(tempcord,
                                            np.array([0.0, -steryang, -sterzang]))

            tempdir1 = modi_rotate_3D_point(np.dot(Qtransfor[i], griddir1[j]),
                                            np.array([0.0, -steryang, -sterzang]))
            tempdir2 = modi_rotate_3D_point(np.dot(Qtransfor[i], griddir2[j]),
                                            np.array([0.0, -steryang, -sterzang]))

            # Then the source coordinates are tilted (due to box geometry).
            trandir1[i*nobeamls + j] = modi_rotate_3D_point(tempdir1,
                                                            np.array([0.0, -normyang, -normzang]))
            trandir2[i*nobeamls + j] = modi_rotate_3D_point(tempdir2,
                                                            np.array([0.0, -normyang, -normzang]))
            # And the source coordinates are tilted and back translated.
            trancord[i*nobeamls + j] = modi_rotate_3D_point(tempcord + turnpoin[i],
                                                            np.array([0.0, -normyang, -normzang])) + qlocations[i]

    return trancord, trandir1, trandir2

#---------------------------------------------------------------------
# Modification routines
#---------------------------------------------------------------------
@nb.njit(nogil=True)
def modi_rotate_x_point(x0: float, a: float):
    """
    Rotates point x0 (x,y,z) around the x axis by an angle a
    (thetax, thetay, thetaz) (a in radians)

    Niek den Harder
    """
    return np.array([x0[0],
                     x0[1] * np.cos(a[0]) - x0[2] * np.sin(a[0]),
                     x0[1] * np.sin(a[0]) + x0[2] * np.cos(a[0])])

@nb.njit(nogil=True)
def modi_rotate_y_point(x0: float, a: float):
    """
    Rotates point x0 (x,y,z) around the y axis by an angle a
    (thetax, thetay, thetaz) (a in radians)

    Niek den Harder
    """
    return np.array([x0[0] * np.cos(a[1]) + x0[2] * np.sin(a[1]),
                     x0[1],
                    -x0[0] * np.sin(a[1]) + x0[2] * np.cos(a[1])])

@nb.njit(nogil=True)
def modi_rotate_z_point(x0: float, a: float):
    """
    Rotates point x0 (x,y,z) around the z axis by an angle a
    (thetax, thetay, thetaz) (a in radians)

    Niek den Harder
    """
    return np.array([x0[0] * np.cos(a[2]) - x0[1] * np.sin(a[2]),
                     x0[0] * np.sin(a[2]) + x0[1] * np.cos(a[2]),
                     x0[2]])

@nb.njit(nogil=True)
def modi_rotate_z_point_list(x0: float, a: float):
    """
    Rotates around the Z axis a list of arrays.

    Niek den Harder
    """
    xr = np.zeros_like(x0)
    for ix, xx in enumerate(x0):
        xr[ix, :] = modi_rotate_z_point(xx, a)
    return xr

@nb.njit(nogil=True)
def modi_rotate_3D_point(x0: float, a: float):
    """
    Rotates point x0 (x,y,z) around the origin by an angle a
    (thetax, thetay, thetaz)

    Niek den Harder
    """
    xr = modi_rotate_x_point(x0, a)
    xr = modi_rotate_y_point(xr, a)
    return modi_rotate_z_point(xr, a)


@nb.njit(nogil=True)
def modi_rotate_3D_point_list(x0: float, a: float):
    """
    Rotates list of points x0 (x,y,z) around the origin by an angle a
    (thetax, thetay, thetaz)

    Niek den Harder
    """
    xr = np.zeros_like(x0)
    for ix, xx in enumerate(x0):
        xr[ix, :] = modi_rotate_3D_point(xx, a)
    return xr


@nb.njit(nogil=True)
def modi_rotate_3D_point_lists(x0: float, a: float):
    """
    Rotates list of points x0 (x,y,z) around the origin by a list of
    angles a (thetax, thetay, thetaz)

    Niek den Harder
    """
    xr = np.zeros_like(x0)
    assert len(x0) == len(a), 'Angles and points must broadcast'
    for ix, xx in enumerate(x0):
        xr[ix, :] = modi_rotate_3D_point(xx, a[ix])
    return xr

@nb.njit(nogil=True)
def modi_transform_box1_to_AUG_cordlist(trancord: float):
    """
    ?

    Niek den Harder
    """
    tokadiff = np.array([2.842*np.cos(33.75*np.pi/180.0),
                         2.842*np.sin(33.75*np.pi/180.0),
                         0.0])
    tokaangl = 18.75*np.pi/180.0
    return modi_rotate_z_point_list(trancord,
                                    np.array([0.0, 0.0, tokaangl])) + tokadiff

@nb.njit(nogil=True)
def modi_transform_box2_to_AUG_cordlist(trancord: float):
    """
    ?

    Niek den Harder
    """
    tokadiff = np.array([3.296*np.cos(29.0*np.pi/180.0),
                         3.296*np.sin(29.0*np.pi/180.0),
                         0.0])
    tokaangl = 10.1*np.pi/180.0

    # This is for the coordinates to agree with the 13/14 old
    # definition of the vessel given in the David file
    tokatran = np.array([-1.0, -1.0, 1.0])
    return (modi_rotate_z_point_list(trancord,
                                     np.array([0.0, 0.0, tokaangl])) + tokadiff)*tokatran

@nb.njit(nogil=True)
def modi_transform_AUG_to_AUG_NEWDEF_cordlist(trancord: float):
    """
    This routine rotates the coordinates of a box,
    so that it is ok with the new AUG definition.
    """
    tokaangl = -(3.0/16.0)*2*np.pi
    return (modi_rotate_z_point_list(trancord,
                                     np.array([0.0, 0.0, tokaangl])))


@nb.njit(nogil=True)
def modi_transform_box1_to_AUG_NEWDEF_cordlist(trancord: float):
    """
    Transform the box 1 coordinates into the AUG new coordinate reference
    frame.

    Niek den Harder
    """
    tempcord = modi_transform_box1_to_AUG_cordlist(trancord)
    return  modi_transform_AUG_to_AUG_NEWDEF_cordlist(tempcord)


@nb.njit(nogil=True)
def modi_transform_box2_to_AUG_NEWDEF_cordlist(trancord: float):
    """
    Transform the box 2 coordinates into the AUG new coordinate reference
    frame.

    Niek den Harder
    """
    tempcord = modi_transform_box2_to_AUG_cordlist(trancord)
    return  modi_transform_AUG_to_AUG_NEWDEF_cordlist(tempcord)

@nb.njit(nogil=True)
def modi_transform_box1_to_AUG_total(trancord: float, trandir1: float,
                                     trandir2: float, trandir3: float):
    """
    Transform the all the box 1 coordinates into the AUG new coordinate
    reference frame.

    Niek den Harder
    """
    tokadiff = np.array([2.842*np.cos(33.75*np.pi/180.0),
                         2.842*np.sin(33.75*np.pi/180.0),
                         0.0])
    tokaangl = 18.75*np.pi/180.0
    tokacord = modi_rotate_z_point_list(trancord,
                                        np.array([0.0, 0.0, tokaangl])) + tokadiff
    tokadir1 = modi_rotate_z_point_list(trandir1,
                                        np.array([0.0, 0.0, tokaangl]))
    tokadir2 = modi_rotate_z_point_list(trandir2,
                                        np.array([0.0, 0.0, tokaangl]))
    tokadir3 = modi_rotate_z_point_list(trandir3,
                                        np.array([0.0, 0.0, tokaangl]))
    return tokacord, tokadir1, tokadir2, tokadir3

@nb.njit(nogil=True)
def modi_transform_box2_to_AUG_total(trancord: float, trandir1: float,
                                     trandir2: float, trandir3: float):
    """
    Transform the all the box 2 coordinates into the AUG new coordinate
    reference frame.

    Niek den Harder
    """
    tokadiff = np.array([3.296*np.cos(29.0*np.pi/180.0),
                         3.296*np.sin(29.0*np.pi/180.0),
                         0.0])
    tokaangl = 10.1*np.pi/180.0

    # This is for the coordinates to agree with the 13/14 old definition of
    # the vessel given in the David file
    tokatran = np.array([-1.0, -1.0, 1.0])
    tokacord = (modi_rotate_z_point_list(trancord,
                                         np.array([0.0, 0.0, tokaangl])) + tokadiff)*tokatran
    tokadir1 = (modi_rotate_z_point_list(trandir1,
                                         np.array([0.0, 0.0, tokaangl])))*tokatran
    tokadir2 = (modi_rotate_z_point_list(trandir2,
                                         np.array([0.0, 0.0, tokaangl])))*tokatran
    tokadir3 = (modi_rotate_z_point_list(trandir3,
                                         np.array([0.0, 0.0, tokaangl])))*tokatran
    return tokacord, tokadir1, tokadir2, tokadir3

@nb.njit(nogil=True)
def modi_transform_AUG_to_AUG_NEWDEF_total(trancord: float, trandir1: float,
                                           trandir2: float, trandir3: float):
    """
    This routine rotates the coordinates of a box,
    so that it is ok with the new AUG definition.

    Niek den Harder
    """
    tokaangl = -(3.0/16.0)*2*np.pi
    tokacord = (modi_rotate_z_point_list(trancord,
                                         np.array([0.0, 0.0, tokaangl])))
    tokadir1 = (modi_rotate_z_point_list(trandir1,
                                         np.array([0.0, 0.0, tokaangl])))
    tokadir2 = (modi_rotate_z_point_list(trandir2,
                                         np.array([0.0, 0.0, tokaangl])))
    tokadir3 = (modi_rotate_z_point_list(trandir3,
                                         np.array([0.0, 0.0, tokaangl])))
    return tokacord, tokadir1, tokadir2, tokadir3


@nb.njit(nogil=True)
def modi_transform_box1_to_AUG_NEWDEF_total(trancord: float, trandir1: float,
                                            trandir2: float, trandir3: float):
    """
    This routine rotates the coordinates of a box,
    so that it is ok with the new AUG definition.

    Niek den Harder
    """
    tempcord, tempdir1, tempdir2, tempdir3 = \
        modi_transform_box1_to_AUG_total(trancord, trandir1, trandir2, trandir3)
    tokacord, tokadir1, tokadir2, tokadir3 = \
        modi_transform_AUG_to_AUG_NEWDEF_total(tempcord, tempdir1,
                                               tempdir2, tempdir3)
    return tokacord, tokadir1, tokadir2, tokadir3


@nb.njit(nogil=True)
def modi_transform_box2_to_AUG_NEWDEF_total(trancord: float, trandir1: float,
                                            trandir2: float, trandir3: float):
    """
    This routine rotates the coordinates of a box,
    so that it is ok with the new AUG definition.

    Niek den Harder
    """
    tempcord, tempdir1, tempdir2, tempdir3 = \
        modi_transform_box2_to_AUG_total(trancord, trandir1, trandir2, trandir3)
    tokacord, tokadir1, tokadir2, tokadir3 = \
        modi_transform_AUG_to_AUG_NEWDEF_total(tempcord, tempdir1,
                                               tempdir2, tempdir3)
    return tokacord, tokadir1, tokadir2, tokadir3


@nb.njit(nogil=True)
def calc_source_cord_normals_AUG_NEWDEF(sourlabl: int,
                                        horiangl: float=0.0,
                                        vertangl: float=0.0):
    """
    Returns the source locations and normals in AUG NEWDEF coordinates
    given a source label and two angles.

    Niek den Harder

    @param sourlabl: source number, that must range in [1, 8] values.
    @param horiangl: horizontal (toroidal) angle wrt the design value.
    @param vertangl: vertical (poloidal) angle wrt the design value.
    """
    beamcord = np.array([[0.0, 0.0, 0.0]])
    beamdir1 = np.array([[0.0, 1.0, 0.0]])
    beamdir2 = np.array([[0.0, 0.0, 1.0]])

    # Preparing the angles for the routines.
    sourangls = np.zeros((4, 2))
    sourangls[:, 0] = horiangl
    sourangls[:, 1] = vertangl

    if sourlabl <= 0:
        raise ValueError('Source value must be positive and larger than 0')

    if (sourlabl <= 4):
        bboxcord, bboxdir1, bboxdir2 = \
            make_beamlets_box1_uwvcords_steer_angle(beamcord, beamdir1,
                                                    beamdir2, sourangls)
        tokacord, tokadir1, tokadir2, tokadir3 = \
            modi_transform_box1_to_AUG_total(bboxcord, bboxdir1,
                                             bboxdir2, np.cross(bboxdir1,
                                                                bboxdir2))
    elif sourlabl <= 8:
        bboxcord, bboxdir1, bboxdir2 = \
            make_beamlets_box2_uwvcords_steer_angle(beamcord, beamdir1,
                                                    beamdir2, sourangls)
        tokacord, tokadir1, tokadir2, tokadir3 = \
            modi_transform_box2_to_AUG_total(bboxcord, bboxdir1,
                                             bboxdir2, np.cross(bboxdir1,
                                                                bboxdir2))
    else:
        raise ValueError('Beam label must range [1, 8]')

    AUGrcord, AUGrdir1, AUGrdir2, AUGrdir3 = \
        modi_transform_AUG_to_AUG_NEWDEF_total(tokacord, tokadir1,
                                               tokadir2, tokadir3)

    retuindx = int(np.mod(sourlabl - 1, 4))

    return AUGrcord[retuindx], AUGrdir3[retuindx]

def get_nbi_cal(source: int, year: int, fn: str='./Data/nbi.yml',
                pos: str='reference'):
    """
    Return the calibration database from the NBI calibration.

    Pablo Oyola - poyola@us.es

    @param source: NBI source to get the calibration.
    @param year: year where the calibration was done.
    @param fn: filename with the calibration database.
    @param pos: position to retrieve. Can only be 'reference', 'on-axis' and
    'off-axis'
    """

    if (source < 1) or (source > 8):
        raise ValueError('The NBI sources in AUG range between 1 and 8')

    if pos.lower() not in ('reference', 'off-axis', 'on-axis'):
        raise ValueError('Input position not valid.')

    with open(fn, 'rt') as fid:
        data = yaml_read(fid).get_data()

    # Rearranging the data into a more useful format.
    years = np.array(list(data.keys()), dtype=int)

    # Get the nearest year.
    found = False
    idx = np.where(years - year < 0)[0][-1]
    while (~found):
        print(idx, found)
        db_year = years[idx]

        # Getting the source calibration.
        name_source = 'Q%d'%source
        if name_source not in data[db_year]:
            if idx == 0:
                return np.array([0.0, 0.0])
            else:
                idx -= 1
                found = False
        else:
            found = True
            break

    # Found calibration.
    if pos.lower() not in data[db_year][name_source]:
        return np.array([0.0, 0.0])

    # Returning the calibration.
    cal = data[db_year][name_source][pos.lower()]
    return np.array([cal['vert'], cal['horz']])

def get_nbi_geom(source: int, lenght: float=15.0, l0: float=0.0,
                 year: int=None):
    """
    Computes the source geometrical parameters for an input beam source.

    Pablo Oyola - poyola@us.es

    @param source: source number, between 1 and 8.
    @param lenght: length from the source origin to build the path of the beam.
    @param l0: length from the source origin to start the beam path.
    """
    if (source <= 0)  or (source > 8):
        raise ValueError('AUG-NBI are labelled between 1 and 8')

    dalpha = 0.0
    dbeta  = 0.0

    if year is None:
        year = 9999
    dalpha, dbeta = get_nbi_cal(source, year)

    # Retrieving the coordinates and the direction of the NBI.
    origin, udir = calc_source_cord_normals_AUG_NEWDEF(source,
                                                       horiangl=dalpha,
                                                       vertangl=dbeta)

    # Computing the main parameters.
    geom = { 'xorigin': origin[0], 'yorigin': origin[1], 'zorigin': origin[2],
             'u': udir, # Cartesian direction vector
             'Rorigin': np.sqrt(origin[0]**2+origin[1]**2),
             'phiorigin': np.arctan2(origin[1], origin[0])
           }

    # --- Tangency radius:
    t = - (origin[1] * udir[1] + origin[0] * udir[0]) / (udir[0]**2 + udir[1]**2)
    geom['xt'] = origin[0] + udir[0] * t
    geom['yt'] = origin[1] + udir[1] * t
    geom['zt'] = origin[2] + udir[2] * t
    geom['rt'] = np.sqrt(geom['xt']**2 + geom['yt']**2)

    # --- Computing starting and ending points.
    r0 = origin + l0*udir
    rend = origin + lenght*udir

    geom.update({ 'x0': r0[0], 'y0': r0[1], 'z0': r0[2],
                  'r0': np.sqrt(r0[0]**2 + r0[1]**2),
                  'phi0': np.arctan2(r0[1], r0[0]),

                  'x1': rend[0], 'y1': rend[1], 'z1': rend[2],
                  'r1': np.sqrt(rend[0]**2 + rend[1]**2),
                  'phi1': np.arctan2(rend[1], rend[0]),
                 })

    # --- Computing the injection angles: beta and alpha.
    geom['alpha'] = np.arctan2(udir[1], udir[0])
    geom['beta']  = np.arctan2(udir[2], np.sqrt(udir[0]**2 + udir[1]**2))

    return geom


