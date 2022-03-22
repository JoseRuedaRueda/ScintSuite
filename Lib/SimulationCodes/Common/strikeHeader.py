"""Contains the headers for the strike points files.

Jose Rueda: jrrueda@us.es

This is the road map. This said to the python library where and how is stored
the data in the files such that the library know where to look.

For each variables it stores:
    - 'i': index of the column where the information is stored
    - 'units': physical units
    - 'longName': long and describing name
    - 'shortName': short name, used to label plots
"""
orderStrikes = {
    'sinpa_INPA': {
        0: {
            'scintillator': {
                'x': {
                    'i': 0,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X Strike position tokamak system',
                    'shortName': 'x',
                },
                'y': {
                    'i': 1,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y Strike position tokamak system',
                    'shortName': 'y',
                },
                'z': {
                    'i': 2,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z Strike position tokamak system',
                    'shortName': 'z'
                },
                'w': {
                    'i': 3,  # Column index in the file
                    'units': ' [au]',  # Units
                    'longName': 'Weight',
                    'shortName': 'Weight',
                },
                'beta': {
                    'i': 4,  # Column index in the file
                    'units': ' [rad]',  # Units
                    'longName': 'beta at pinhole',
                    'shortName': '$\\beta$',
                },
                'xi0': {
                    'i': 5,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X initial tokamak system',
                    'shortName': '$x_{i}$',
                },
                'yi0': {
                    'i': 6,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y initial tokamak system',
                    'shortName': '$y_{i}$',
                },
                'zi0': {
                    'i': 7,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z initial tokamak system',
                    'shortName': '$z_{i}$',
                },
                'xs': {
                    'i': 8,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X strike scintillator system',
                    'shortName': '$x_{s}$',
                },
                'ys': {
                    'i': 9,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y strike scintillator system',
                    'shortName': '$y_{s}$',
                },
                'zs': {
                    'i': 10,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z strike scintillator system',
                    'shortName': '$z_{s}$',
                },
                'theta': {
                    'i': 11,  # Column index in the file
                    'units': ' [deg]',  # Units
                    'longName': 'Incident angle on the scintillator',
                    'shortName': '$\\phi_i$',
                },
                'xnbi': {
                    'i': 12,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X position of the closest point to NBI line',
                    'shortName': '$x_{NBI}$',
                },
                'ynbi': {
                    'i': 13,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y position of the closest point to NBI line',
                    'shortName': '$y_{NBI}$',
                },
                'znbi': {
                    'i': 14,  # Column index in the file
                    'units': ' [cm]',  # Units
                    'longName': 'Z position of the closest point to NBI line',
                    'shortName': '$z_{NBI}$',
                },
                'vx0': {
                    'i': 15,  # Column index in the file
                    'units': ' [m/s]',  # Units
                    'longName': 'vx at pinhole',
                    'shortName': '$v_{x0}$',
                },
                'vy0': {
                    'i': 16,  # Column index in the file
                    'units': ' [m/s]',  # Units
                    'longName': 'vy at pinhole',
                    'shortName': '$v_{y0}$',
                },
                'vz0': {
                    'i': 17,  # Column index in the file
                    'units': ' [m/s]',  # Units
                    'longName': 'vz at pinhole',
                    'shortName': '$v_{z0}$',
                },
                'dmin': {
                    'i': 18,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Minimum distance to NBI',
                    'shortName': '$d_{min}$',
                },
            },   # Mapping version 0
            'collimator': {
                'x': {
                    'i': 0,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X Strike position tokamak system',
                    'shortName': 'x',
                },
                'y': {
                    'i': 1,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y Strike position tokamak system',
                    'shortName': 'y',
                },
                'z': {
                    'i': 2,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z Strike position tokamak system',
                    'shortName': 'z'
                },
                'w': {
                    'i': 3,  # Column index in the file
                    'units': ' [au]',  # Units
                    'longName': 'Weight',
                    'shortName': 'Weight',
                },
            },  # End of colimator version 0
            'signalscintillator': {
                'FIDASIMid': {
                    'i': 0,  # Column index in the file
                    'units': '',  # Units
                    'longName': 'FIDASIM id',
                    'shortName': 'id',
                },
                'x': {
                    'i': 1,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X Strike position tokamak system',
                    'shortName': 'x',
                },
                'y': {
                    'i': 2,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y Strike position tokamak system',
                    'shortName': 'y',
                },
                'z': {
                    'i': 3,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z Strike position tokamak system',
                    'shortName': 'z'
                },
                'xs': {
                    'i': 4,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X strike scintillator system',
                    'shortName': '$x_{s}$',
                },
                'ys': {
                    'i': 5,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y strike scintillator system',
                    'shortName': '$y_{s}$',
                },
                'zs': {
                    'i': 6,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z strike scintillator system',
                    'shortName': '$z_{s}$',
                },
                'xcx': {
                    'i': 7,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X of CX reaction',
                    'shortName': '$x_{CX}$',
                },
                'ycx': {
                    'i': 8,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y of CX reaction',
                    'shortName': '$y_{CX}$',
                },
                'zcx': {
                    'i': 9,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z of CX reaction',
                    'shortName': '$z_{CX}$',
                },
                'vx0': {
                    'i': 10,  # Column index in the file
                    'units': ' [m/s]',  # Units
                    'longName': 'vx at pinhole',
                    'shortName': '$v_{x0}$',
                },
                'vy0': {
                    'i': 11,  # Column index in the file
                    'units': ' [m/s]',  # Units
                    'longName': 'vy at pinhole',
                    'shortName': '$v_{y0}$',
                },
                'vz0': {
                    'i': 12,  # Column index in the file
                    'units': ' [m/s]',  # Units
                    'longName': 'vz at pinhole',
                    'shortName': '$v_{z0}$',
                },
                'w': {
                    'i': 13,  # Column index in the file
                    'units': ' [part/s/cm^2 pinhole]',  # Units
                    'longName': 'Weight',
                    'shortName': 'Weight',
                },
                'kind': {
                    'i': 14,  # Column index in the file
                    'units': '',  # Units
                    'longName': 'Marker kind',
                    'shortName': '$k$',
                },
            },   # End of INPA signal version 0
        }
    },
    'sinpa_FILD': {
        0: {
            'scintillator': {
                'x': {
                    'i': 0,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X Strike position tokamak system',
                    'shortName': 'x',
                },
                'y': {
                    'i': 1,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y Strike position tokamak system',
                    'shortName': 'y',
                },
                'z': {
                    'i': 2,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z Strike position tokamak system',
                    'shortName': 'z'
                },
                'w': {
                    'i': 3,  # Column index in the file
                    'units': ' [au]',  # Units
                    'longName': 'Weight',
                    'shortName': 'Weight',
                },
                'beta': {
                    'i': 4,  # Column index in the file
                    'units': ' [rad]',  # Units
                    'longName': 'Gyrophase',
                    'shortName': '$\\beta$',
                },
                'xi0': {
                    'i': 5,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X initial tokamak system',
                    'shortName': '$x_{i}$',
                },
                'yi0': {
                    'i': 6,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y initial tokamak system',
                    'shortName': '$y_{i}$',
                },
                'zi0': {
                    'i': 7,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z initial tokamak system',
                    'shortName': '$z_{i}$',
                },
                'xs': {
                    'i': 8,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X strike scintillator system',
                    'shortName': '$x_{s}$',
                },
                'ys': {
                    'i': 9,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y strike scintillator system',
                    'shortName': '$y_{s}$',
                },
                'zs': {
                    'i': 10,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z strike scintillator system',
                    'shortName': '$z_{s}$',
                },
                'theta': {
                    'i': 11,  # Column index in the file
                    'units': ' [deg]',  # Units
                    'longName': 'Incident angle on the scintillator',
                    'shortName': '$\\phi_i$',
                },
            },
            'collimator': {
                'x': {
                    'i': 0,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'X Strike position tokamak system',
                    'shortName': 'x',
                },
                'y': {
                    'i': 1,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Y Strike position tokamak system',
                    'shortName': 'y',
                },
                'z': {
                    'i': 2,  # Column index in the file
                    'units': ' [m]',  # Units
                    'longName': 'Z Strike position tokamak system',
                    'shortName': 'z'
                },
                'w': {
                    'i': 3,  # Column index in the file
                    'units': ' [au]',  # Units
                    'longName': 'Weight',
                    'shortName': 'Weight',
                },
            },
        }  # End of FILD MODE version 0
    },
    'fildsim_FILD': {
        0: {
            'beta': {
                'i': 0,  # Column index in the file
                'units': ' [rad]',  # Units
                'longName': 'Initial gyrophase',
                'shortName': '$\\beta$',
            },
            'x': {
                'i': 1,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X Strike position',
                'shortName': 'x',
            },
            'y': {
                'i': 2,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y Strike position',
                'shortName': 'y',
            },
            'z': {
                'i': 3,  # Column index in the file
                'units': ' [cm]',   # Units
                'longName': 'Z Strike position',
                'shortName': 'z'
            },
            'remap_rl': {
                'i': 4,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Remapped Larmor radius',
                'shortName': '$r_l$',
            },
            'remap_XI': {
                'i': 5,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Remapped pitch angle',
                'shortName': '$\\lambda$',
            },
            'remap_pitch': {   # Repeated for retrocompatibility
                'i': 5,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Remapped pitch angle',
                'shortName': '$\\lambda$',
            },
            'theta': {
                'i': 6,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Incident angle',
                'shortName': '$\\phi$',
            },
        },  # End of version 0
        1: {
            'beta': {
                'i': 0,  # Column index in the file
                'units': ' [rad]',  # Units
                'longName': 'Initial gyrophase',
                'shortName': '$\\beta$',
            },
            'x': {
                'i': 1,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X Strike position',
                'shortName': 'x',
            },
            'y': {
                'i': 2,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y Strike position',
                'shortName': 'y',
            },
            'z': {
                'i': 3,  # Column index in the file
                'units': ' [cm]',   # Units
                'longName': 'Z Strike position',
                'shortName': 'z'
            },
            'remap_rl': {
                'i': 4,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Remapped Larmor radius',
                'shortName': '$r_l$',
            },
            'remap_XI': {
                'i': 5,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Remapped pitch angle',
                'shortName': '$\\lambda$',
            },
            'remap_pitch': {   # Repeated for retrocompatibility
                'i': 5,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Remapped pitch angle',
                'shortName': '$\\lambda$',
            },
            'theta': {
                'i': 6,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Incident angle',
                'shortName': '$\\phi$',
            },
            'xi0': {
                'i': 7,   # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X initial position',
                'shortName': '$x_{i}$',
            },
            'yi0': {
                'i': 8,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y initial position',
                'shortName': '$y_{i}$',
            },
            'zi0': {
                'i': 9,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Z initial position',
                'shortName': '$z_{i}$',
            },  # End of version 1
        },
    }
}
