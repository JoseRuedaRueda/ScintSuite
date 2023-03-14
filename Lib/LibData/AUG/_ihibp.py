"""
Routines to calculate the impact of the deflection plates on the ihibp beam.
Returns the new injection angles beta, theta as well as the new point of origin.

hannah.lindl@ipp.mpg.de
"""

import numpy as np
from scipy.constants import elementary_charge as ec
# import Lib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
logger = logging.getLogger('aug_sfutils')


class DeflectionPlates():

    def __init__(self, Ua, tor_pos: float = 0.0, tor_neg:
                 float = 0.0, pol_pos: float = 0.0, pol_neg: float = 0.0):
        """
        calculate the effect of the deflection plates on the beamline
        :param beta_old: toroidal angle w/o deflection
        :param theta_old: poloidal angle w/o deflection
        :param Ua: beam acceleration voltage in kV
        :param tor_pos:: positive toroidal plate voltage
        :param tor_neg:: negative toroidal plate voltage
        :param pol_pos:: positive poloidal plate voltage
        :param pol_neg:: negative poloidal plate voltage

        hannah.lindl@ipp.mpg.de

        """
        Ua *= 1e3  #voltage in [V]

        self.deflection_plates = {'l': 0.104, 'deltax': 0.046,
                                  'plane_point_tor': [0.848, -4.909, 0.03],
                                  'plane_point_pol': [0.830, -4.748, 0.03]}

        self.old_beam = {'m': 1.41e-25,
                         'port_center': [0.681, -3.457,  0.027],
                         'emitter': [0.888, -5.277, 0.03], 'Ua': Ua}

        v0 = np.sqrt(2*ec*self.old_beam['Ua']/self.old_beam['m'])
        self.old_beam['v0'] = v0

        deltaV_tor = tor_pos - tor_neg
        deltaV_pol = pol_pos - pol_neg

        if deltaV_tor != 0:
            self.deflection_plates['deltaV_tor'] = deltaV_tor
        if deltaV_pol != 0:
            self.deflection_plates['deltaV_pol'] = deltaV_pol

        #beam vector
        b0_vec = np.array(self.old_beam['port_center']) - \
                 np.array(self.old_beam['emitter'])
        b0_vec /= np.linalg.norm(b0_vec)
        v0_vec = v0*b0_vec

        self.deflection_plates['tor_defl_point'], \
                    b1_vec, v1_vec = self.deflection(b0_vec, v0_vec,
                                                     name = 'tor')
        self.deflection_plates['pol_defl_point'], \
                    b2_vec, v2_vec = self.deflection(b1_vec, v1_vec,
                                                     name = 'pol')

        # new origin: intersection of radial plane through old vacuum port
        # position with new beamline passing through pol_defl_point
        new_origin = LinePlaneCollision(np.array([1,1,0]),
                                        self.old_beam['port_center'], b2_vec,
                                        self.deflection_plates['pol_defl_point'])
        self.beam = {'port_center': new_origin, 'final_beam': b2_vec}
        self.beam['beta'], self.beam['theta'] = self.angles()
        self.beam['v0'] = v0_vec
        self.beam['v1'] = v1_vec
        self.beam['v2'] = v2_vec

        print(repr(self))
        return # params to Lib.params()

    def deflection(self, beam, v_vec, name: str = ''):
        if name == 'tor':
            defl_point = LinePlaneCollision(beam,
                                    self.deflection_plates['plane_point_tor'],
                                    beam, self.old_beam['emitter'])
            if 'deltaV_tor' in self.deflection_plates:
                F = ec*self.deflection_plates['deltaV_tor']/\
                    self.deflection_plates['deltax']
                n_vec = np.array([-beam[1], beam[0], 0])
                n_vec /= np.linalg.norm(n_vec)
            else:
                logger.info('no toroidal voltage. beam not deflected toroidally')
                return defl_point, beam, v_vec

        elif name == 'pol':
            defl_point = LinePlaneCollision(beam,
                                            self.deflection_plates['plane_point_pol'],
                                            beam,
                                            self.deflection_plates['tor_defl_point'])
            if 'deltaV_pol' in self.deflection_plates:
                F = ec*self.deflection_plates['deltaV_pol']/\
                    self.deflection_plates['deltax']
                """
                deflection force acts perpendicular to beam:
                b = [rcos(phi), rsin(phi), z]
                n = [-z*cos(phi), -z*sin(phi), r]
                """
                rho, phi = cart2pol(defl_point[0], defl_point[1])
                nx = -beam[2]*np.cos(phi)
                ny = -beam[2]*np.sin(phi)
                nz = rho
                n_vec = -np.array([nx, ny, nz])
                n_vec /= np.linalg.norm(n_vec)
            else:
                logger.info('no poloidal voltage. beam not deflected poloidally')
                return defl_point, beam, v_vec

        else:
            logger.info('type not recognized. should be tor or pol')

        a = F/self.old_beam['m']
        v_norm = np.linalg.norm(v_vec)
        time = self.deflection_plates['l']/v_norm
        a_vec = a*n_vec

        v1_vec = v_vec + a_vec*time
        # v1_norm = np.linalg.norm(v1_vec)
        # v1_vec /= v1_norm

        b1_vec = defl_point + v1_vec
        b1_vec /= np.linalg.norm(b1_vec)

        return defl_point, b1_vec, v1_vec

    def angles(self):
        #beta: angle between x and y beam coordinate
        orig = self.beam['port_center']
        beam = self.beam['final_beam']
        cosb = np.sum(beam[:2]*orig[:2])/(np.linalg.norm(beam[:2])*np.linalg.norm(orig[:2]))
        beta = (180 - np.rad2deg(np.arccos(cosb)))

        #theta: angle between r and z beam coordinate
        beam_rz = np.array([np.sqrt(beam[0]**2+beam[1]**2), beam[2]])
        origr = np.array([np.sqrt(orig[0]**2+orig[1]**2), 0])
        cost = np.sum(beam_rz*origr)/(np.linalg.norm(beam_rz)*np.linalg.norm(origr))
        theta = np.rad2deg(np.arccos(cost))

        return beta, theta


    def plot(self, ax = None):

        if ax == None:
            ax_was_none = True
            ax = Axes3D(plt.figure())

        emit = self.old_beam['emitter']
        vac_old = self.old_beam['port_center']
        vac_new = self.beam['port_center']
        tor = self.deflection_plates['tor_defl_point']
        pol = self.deflection_plates['pol_defl_point']
        tor_old = self.deflection_plates['plane_point_tor']
        pol_old = self.deflection_plates['plane_point_pol']

        t = np.linspace(0, 5e-6) # time axis

        x0 = emit[0] + self.beam['v0'][0]*t
        y0 = emit[1] + self.beam['v0'][1]*t
        z0 = emit[2] + self.beam['v0'][2]*t

        x1 = tor[0] + self.beam['v1'][0]*t
        y1 = tor[1] + self.beam['v1'][1]*t
        z1 = tor[2] + self.beam['v1'][2]*t

        x2 = pol[0] + self.beam['v2'][0]*t
        y2 = pol[1] + self.beam['v2'][1]*t
        z2 = pol[2] + self.beam['v2'][2]*t

        ax.scatter3D(*emit, label = 'emitter')
        ax.scatter3D(*vac_old, label = 'vacuum port')
        ax.scatter3D(*vac_new, label = 'new injection point')
        ax.scatter3D(*tor, label = 'tor_defl_point')
        ax.scatter3D(*pol, label = 'pol_defl_point')
        ax.scatter3D(*tor_old, label = 'tor_defl_plate')
        ax.scatter3D(*pol_old, label = 'pol_defl_plate')
        # ax.plot3D([emit[0], beam0[0]], [emit[1], beam0[1]],
        #           [emit[2], beam0[2]], label='beam_0')
        # ax.plot3D([tor[0], beam1[0]], [tor[1], beam1[1]],
        #           [tor[2], beam1[2]], label='beam_tor')
        # ax.plot3D([pol[0], beam2[0]], [pol[1], beam2[1]],
        #           [pol[2], beam2[2]], label='beam_pol')
        ax.plot3D(x0, y0, z0, label='beam0')
        ax.plot3D(x1, y1, z1, label='beam_tor')
        ax.plot3D(x2, y2, z2, label='beam_pol')

        if ax_was_none:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            # ax.set_xlim([0.6, 0.9])
            # ax.set_ylim([-6,-2])
            # ax.set_zlim([0.026,0.03])

        plt.legend()


    def __repr__(self):
        return 'theta =  %.2f, beta = %.2f, origin = ' %(self.beam['theta'],
                self.beam['beta']) +  str(self.beam['port_center'])



def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    """
    calculate the point where the beamline passes through the deflection
    plates
    """
    planePoint = np.array(planePoint)
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    point = w + si * rayDirection + planePoint
    return point

def cart2pol(x,y):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    return rho, phi
