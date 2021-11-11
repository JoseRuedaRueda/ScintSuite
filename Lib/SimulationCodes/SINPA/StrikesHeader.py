"""
Contains the headers for the SINPA strike points files
"""
order_INPA = {
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
            'inc_alpha': {
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
        'signalscintillator': {
            'FIDASIMid': {
                'i': 0,  # Column index in the file
                'units': '',  # Units
                'longName': 'FIDASIM id',
                'shortName': 'id',
            },
            'x': {
                'i': 1,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X Strike position tokamak system',
                'shortName': 'x',
            },
            'y': {
                'i': 2,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y Strike position tokamak system',
                'shortName': 'y',
            },
            'z': {
                'i': 3,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Z Strike position tokamak system',
                'shortName': 'z'
            },
            'xs': {
                'i': 4,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X strike scintillator system',
                'shortName': '$x_{s}$',
            },
            'ys': {
                'i': 5,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y strike scintillator system',
                'shortName': '$y_{s}$',
            },
            'zs': {
                'i': 6,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Z strike scintillator system',
                'shortName': '$z_{s}$',
            },
            'xi': {
                'i': 7,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X of CX reaction',
                'shortName': '$x_{CX}$',
            },
            'yi': {
                'i': 8,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y of CX reaction',
                'shortName': '$y_{CX}$',
            },
            'zi': {
                'i': 9,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Z of CX reaction',
                'shortName': '$z_{CX}$',
            },
            'vx0': {
                'i': 10,  # Column index in the file
                'units': ' [cm/s]',  # Units
                'longName': 'vx at pinhole',
                'shortName': '$v_{x0}$',
            },
            'vy0': {
                'i': 11,  # Column index in the file
                'units': ' [cm/s]',  # Units
                'longName': 'vy at pinhole',
                'shortName': '$v_{y0}$',
            },
            'vz0': {
                'i': 12,  # Column index in the file
                'units': ' [cm/s]',  # Units
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
            'beta': {
                'i': 11,  # Column index in the file
                'units': ' [rad]',  # Units
                'longName': 'beta at pinhole',
                'shortName': '$\\beta$',
            },
        },
    }
}


order_FILD = {
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
            'inc_alpha': {
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
    }
}
