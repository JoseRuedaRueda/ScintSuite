"""Hydrogen and lithium ionization formulae"""

import numpy as np
import Lib.LibParameters as sspar
import Lib.LibPlotting as ssplt
import matplotlib.pyplot as plt

# Importing libraries with the Cs and the rubidium cross sections.
# import netCDF4 as nc4

from scipy.interpolate import interp1d
from scipy.integrate import tplquad, quad
from scipy.special import kn as modBessel2nd
from tqdm import tqdm
from datetime import date
from scipy.io import savemat

# ----------------------------------------------------------
# --- Binding energy for different alkali atoms.
# ----------------------------------------------------------

H0_1s1  = 13.598433  # Ionization energy of Hydrogen in ground state.
Li0_2s1 = 5.391719  # First ionization energy of Lithium in ground state.
LiI_1s2 = 75.6400  # Second ionization energy of Lithium (Li+ in GS)
Cs0_6s1 = 3.89390   # First ionization energy of Cs0 (Ground state)
CsI_5p6 = 12.3  # Second ionization energy of Cs (Ground state of Cs+)
Rb0_5s1 = 4.17713  # First ionization energy of Rb0 (Ground state)
RbI_4p6 = 15.3  # Second ionization energy of Rb0 (Ground state of Rb+)

# ----------------------------------------------------------
# --- Different masses of Alkali atoms in AMU.
# ----------------------------------------------------------
alkMasses = {
    'H': 1.00782503223,
    'D': 2.01410177812,
    'T': 3.0160492779,
    'Li6': 6.0151228874,
    'Li7': 7.0160034366,
    'Li': 7.0160034366,
    'C12': 12,
    'C': 12,
    'B10': 10.01293695,
    'B11': 11.00930536,
    'B': 11.00930536,
    'Rb85': 84.9117897379,
    'Rb87': 86.9091805310,
    'Cs133': 132.9054519610,
    'Cs': 132.9054519610
}

reducedMasses = {
    'Cs_H': alkMasses['H'] * alkMasses['Cs']/(alkMasses['H'] + alkMasses['Cs']),
    'Cs_D': alkMasses['D'] * alkMasses['Cs']/(alkMasses['D'] + alkMasses['Cs']),
    'Rb85_H': alkMasses['H'] * alkMasses['Rb85']/(alkMasses['H'] + alkMasses['Rb85']),
    'Rb85_D': alkMasses['D'] * alkMasses['Rb85']/(alkMasses['D'] + alkMasses['Rb85']),
    'Rb87_H': alkMasses['H'] * alkMasses['Rb87']/(alkMasses['H'] + alkMasses['Rb85']),
    'Rb87_D': alkMasses['D'] * alkMasses['Rb87']/(alkMasses['D'] + alkMasses['Rb85'])
}
# ----------------------------------------------------------
# --- Hydrogen ionization cross-section.
# ----------------------------------------------------------

"""
Data for Hydrogen from
R. K. Janev and J. J. Smith, Nucl. Fusion A / M
Suppl. 4, 1 (1993)

https://inis.iaea.org/collection/NCLCollectionStore/_Public/25/024/25024274.pdf?r=1&r=1

sigma in cm**2
"""

def H0_proton_imp_ion(E):
    """
    Analytic formula in page 68.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy per amu where cross-section needs to be evaluated.
    @return sigma: cross-section in [cm^2]
    """

    A1 = 12.899
    A2 = 61.897
    A3 = 9.2731e3
    A4 = 4.9749e-4
    A5 = 3.9890e-2
    A6 = -1.5900
    A7 = 3.1834
    A8 = -3.7154

    factor = 1.0e-16*A1

    sigma = np.exp(-A2/E)*np.log(1+A3*E)/E
    sigma += A4*np.exp(-A5*E)/(np.power(E, A6) + A7 * np.power(E, A8))

    sigma = sigma*factor;

    output = {}
    output['base']  = E
    output['base_units'] = 'eV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0 ion impact ionization cross-section'
    output['name_short'] = 'H-H impact'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Hydrogen',
                        'symbol': 'H^0',
                        'binding_energy': H0_1s1,
                        'binding_unit': 'eV'
                     }

    for ii in range(len(E)):
        if E[ii] < 0.2e3:
            output['uncertainty'][ii] = 1.00
        elif E[ii] < 2e3:
            output['uncertainty'][ii] = 0.20
        elif E[ii] < 5.0e3:
            output['uncertainty'][ii] = 0.30
        elif E[ii] < 10e3:
            output['uncertainty'][ii] = 0.20
        else:
            output['uncertainty'][ii] = 0.10

    return output

def H0_proton_CX(E):
    """
    Analytic formula in page 78.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy per amu where cross-section needs to be evaluated.
    @return sigma: cross-section in [cm^2]
    """

    A1 = 3.2345
    A2 = 235.88
    A3 = 0.038371
    A4 = 3.8068e-6
    A5 = 1.1832e-10
    A6 = 2.3713

    factor = 1e-16*A1

    sigma = np.log(A2/E + A6)
    sigma /= 1.0 + A3*E + A4*np.power(E, 3.5) + A5*np.power(E, 5.4)
    sigma *= factor


    output = {}
    output['base']  = E
    output['base_units'] = 'eV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0 ion CX ionization cross-section'
    output['name_short'] = 'H CX cross-section'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Hydrogen',
                        'symbol': 'H^0',
                        'binding_energy': H0_1s1,
                        'binding_unit': 'eV'
                     }

    for ii in range(len(E)):
        if E[ii] < 10:
            output['uncertainty'][ii] = 0.10
        elif E[ii] < 1000:
            output['uncertainty'][ii] = 0.15
        elif E[ii] < 1.0e5:
            output['uncertainty'][ii] = 0.10
        elif E[ii] < 2.0e6:
            output['uncertainty'][ii] = 0.20
        elif E[ii] < 1e7:
            output['uncertainty'][ii] = 0.40
        else:
            output['uncertainty'][ii] = 1.00

    return output

def H0_Boron_impact(E):
    """
    Boron-Hydrogen impact ionization cross section (Page 152)

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy in the CM frame [eV]
    @return output: dictionary with the evaluated cross section.
    """
    A1 = 351.52
    A2 = 233.63
    A3 = 3.2952e3
    A4 = 5.3787e-6
    A5 = 1.8834e-2
    A6 = -2.2064
    A7 = 7.2074
    A8 = -3.78664

    factor = 1.0e-16*A1

    sigma = np.exp(-A2/E)*np.log(1+A3*E)/E
    sigma += A4*np.exp(-A5*E)/(np.power(E, A6) + A7 * np.power(E, A8))
    sigma *= factor


    output = {}
    output['base']  = E
    output['base_units'] = 'keV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0-B5 impact ionization cross-section'
    output['name_short'] = 'H-B5 cross-section'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Hydrogen',
                        'symbol': 'H^0',
                        'binding_energy': H0_1s1,
                        'binding_unit': 'eV'
                     }

    for ii in range(len(E)):
        if E[ii] < 5:
            output['uncertainty'][ii] = 1.00
        elif E[ii] < 80:
            output['uncertainty'][ii] = 0.80
        elif E[ii] < 200:
            output['uncertainty'][ii] = 0.40
        elif E[ii] < 400:
            output['uncertainty'][ii] = 0.30
        else:
            output['uncertainty'][ii] = 0.15

    return output

def H0_Carbon_impact(E):
    """
    Boron-Hydrogen impact ionization cross section (Page 152)

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy in the CM frame [eV]
    @return output: dictionary with the evaluated cross section.
    """
    A1 = 438.36
    A2 = 327.10
    A3 = 1.4444e5
    A4 = 3.5212e-3
    A5 = 8.3031e-3
    A6 = -0.63731
    A7 = 1.9116e4
    A8 = -3.1003

    factor = 1.0e-16*A1

    sigma = np.exp(-A2/E)*np.log(1+A3*E)/E
    sigma += A4*np.exp(-A5*E)/(np.power(E, A6) + A7 * np.power(E, A8))
    sigma *= factor


    output = {}
    output['base']  = E
    output['base_units'] = 'keV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0-C6 impact ionization cross-section'
    output['name_short'] = 'H-C6 cross-section'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Hydrogen',
                        'symbol': 'H^0',
                        'binding_energy': H0_1s1,
                        'binding_unit': 'eV'
                     }

    for ii in range(len(E)):
        if E[ii] < 10:
            output['uncertainty'][ii] = 1.00
        elif E[ii] < 60:
            output['uncertainty'][ii] = 0.80
        elif E[ii] < 200:
            output['uncertainty'][ii] = 0.40
        elif E[ii] < 400:
            output['uncertainty'][ii] = 0.30
        else:
            output['uncertainty'][ii] = 0.15

    return output

def H0_B5_CX(E):
    """
    Hydrogen charge-exchange reaction cross-section.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy in the CM frame [eV]
    @return output: dictionary with the evaluated cross section.
    """
    A1 = 31.226
    A2 = 1.1442
    A3 = 4.8372e-8
    A4 = 3.0961e-10
    A5 = 4.7205
    A6 = 6.2844e-7
    A7 = 3.1297
    A8 = 0.12556
    A9 = 0.30098
    A10 = 5.9607e-2
    A11 = -0.57923

    factor = 1.0e-16*A1

    sigma = np.exp(-A2/np.power(E, A8))
    sigma /= 1.0 + A3*np.power(E, 2.0) + A4*np.power(E, A5) \
           + A6*np.power(E, A7)
    sigma += A9*np.exp(-A10*E)/np.power(E, A11)
    sigma *= factor

    output = {}
    output['base']  = E
    output['base_units'] = 'keV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0-B5 CX ionization cross-section'
    output['name_short'] = 'H-B5 CX'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Hydrogen',
                        'symbol': 'H^0',
                        'binding_energy': H0_1s1,
                        'binding_unit': 'eV'
                     }

    for ii in range(len(E)):
        if E[ii] < 0.1:
            output['uncertainty'][ii] = 1.00
        elif E[ii] < 10:
            output['uncertainty'][ii] = 0.20
        elif E[ii] < 80:
            output['uncertainty'][ii] = 0.25
        else:
            output['uncertainty'][ii] = 0.15

    return output

def H0_C6_CX(E):
    """
    Hydrogen charge-exchange reaction cross-section.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy in the CM frame [eV]
    @return output: dictionary with the evaluated cross section.
    """
    A1 = 418.18
    A2 = 2.1585
    A3 = 3.4808e-4
    A4 = 5.3333e-9
    A5 = 4.6556
    A6 = 0.33755
    A7 = 0.81736
    A8 = 0.27874
    A9 = 1.8003e-3
    A10 = 7.1033e-2
    A11 = 0.53261

    factor = 1.0e-16*A1

    sigma = np.exp(-A2/np.power(E, A8))
    sigma /= 1.0 + A3*np.power(E, 2.0) + A4*np.power(E, A5) \
           + A6*np.power(E, A7)
    sigma += A9*np.exp(-A10*E)/np.power(E, A11)
    sigma *= factor

    output = {}
    output['base']  = E
    output['base_units'] = 'keV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0-C6 CX ionization cross-section'
    output['name_short'] = 'H-C6 CX'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Hydrogen',
                        'symbol': 'H^0',
                        'binding_energy': H0_1s1,
                        'binding_unit': 'eV'
                     }

    for ii in range(len(E)):
        if E[ii] < 0.05:
            output['uncertainty'][ii] = 1.00
        elif E[ii] < 1:
            output['uncertainty'][ii] = 0.30
        elif E[ii] < 100:
            output['uncertainty'][ii] = 0.15
        else:
            output['uncertainty'][ii] = 0.20

    return output

# ----------------------------------------------------------
# --- Lithium ionization cross-section.
# ----------------------------------------------------------
def Li_proton_CX_ion(E):
    """
    Lithium ionization via charge-exchange with hydrogen.

    Data from:
    D. Wutte et al., ATOMIC DATA AND NUCLEAR DATA TABLES65,155 * 180 (1997),
    https://www.sciencedirect.com/science/article/pii/S0092640X97907361

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy per amu where cross-section needs to be evaluated.
    @return sigma: cross-section in [cm^2]
    """
    A1 = 5.7260e-18
    A2 = 0.2774610534
    A3 = -0.9980483025
    A4 = 0.005285640316
    A5 = 0.002786487304
    A6 = 0.6859035811
    A7 = 553.7383722e2
    A8 = 1.998895199
    A9 = 1.579992279
    A10 = 7.521161583e-7


    sigma  = np.exp(-A2/E)*np.log(2.718+A10/E)/(1.0 + A3*np.power(E, A4) \
                                                   + A5*np.power(E, A6))
    sigma += A7*np.exp(-A8/E)*(A9 + np.log(E))/E
    sigma *= A1


    output = {}
    output['base']  = E
    output['base_units'] = 'eV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'Li^0 CX impact ionization cross-section'
    output['name_short'] = 'Li-H impact'
    output['uncertainty'] = np.zeros(len(E))
    output['atom'] = {
                        'name': 'Lithium',
                        'symbol': 'Li^0',
                        'binding_energy': Li0_2s1,
                        'binding_unit': 'eV'
                     }

    return output

def Li_proton_imp_ion(E):
    """
    Cross section for the hydrogen impact ionization with Li:
        Li + H = Li + H + e

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    Analytic formula in page 68.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param E: energy per amu where cross-section needs to be evaluated.
    @return output: dictionary with the evaluated cross section data.
    """

    A1 = 17.401
    A2 = 6.5623
    A3 = 7.1943e6
    A4 = 67.115
    A5 = 0.21151
    A6 = 1.0551
    A7 = 64.854
    A8 = -1.4472

    factor = 1.0e-16*A1

    sigma = np.exp(-A2/E)*np.log(1+A3*E)/E
    sigma += A4*np.exp(-A5*E)/(np.power(E, A6) + A7 * np.power(E, A8))

    sigma = sigma*factor;

    output = {}
    output['base']  = E
    output['base_units'] = 'eV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'H^0 ion impact ionization cross-section'
    output['name_short'] = 'Li-H impact'
    output['atom'] = {
                        'name': 'Lithium',
                        'symbol': 'Li^0',
                        'binding_energy': Li0_2s1,
                        'binding_unit': 'eV'
                     }
    return output

def Li_heavy_imp_ion(E, q, Ub: float = Li0_2s1):
    """
    Heavy-ion to lithium charge-exchange reactions.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    Data take from:
    D. Wutte et al., ATOMIC DATA AND NUCLEAR DATA TABLES65,155 * 180 (1997),
    https://www.sciencedirect.com/science/article/pii/S0092640X97907361

    @param E: energy where the cross section is going to be evaluated.
    @param q: charge state of the heavy-ion with which the Li atom
    will collide.
    @param Ub: binding energy of the atom which gets ionized.
    @return output: dictionary with the cross section evaluated.
    """

    A1 = 7.513e-10
    A2 = 3.14
    A3 = 8.307e5
    A4 = 1.41e-4
    A5 = 6.062

    # Reduced energy calculation.
    neff  = np.sqrt(H0_1s1/Ub)
    Ered  = neff**2/np.sqrt(q)*E

    sigma  = 1 - np.exp(-np.power(Ered, A2)/A3)
    sigma /= np.power(Ered, A2) + A4*np.power(Ered, A5)
    sigma *= A1

    sigma *= neff**4 * q

    output = {}
    output['base']  = E
    output['base_units'] = 'eV/amu'
    output['base_name'] = 'Energy'
    output['base_short'] = 'E'
    output['base_type'] = 'e'
    output['sigma'] = sigma
    output['units'] = '$cm^2$'
    output['name_long'] = 'Li^0 heavy-ion impact ionization cross-section'
    output['name_short'] = 'Li-A^q impact'
    output['atom'] = {
                        'name': 'Lithium',
                        'symbol': 'Li^0',
                        'binding_energy': Li0_2s1,
                        'binding_unit': 'eV'
                        ''
                     }
    return output

# ----------------------------------------------------------
#--- Scaling relation
# ----------------------------------------------------------
def Wutte_scaling(xsection: dict, Ubnew: float):
    """
    Scales the cross-section of a provided input to a new atom.
    It is strongly expected that this scaling works better when the same group
    of atoms are considered, i.e., among alkali and so on.

    Scaling taken from:
    D. Wutte et al., ATOMIC DATA AND NUCLEAR DATA TABLES65,155 * 180 (1997),
    https://www.sciencedirect.com/science/article/pii/S0092640X97907361


    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param xsection: dictionary with all the cross-section data, including
    the energy where it is evaluated and the original atom.
    @param Ubnew: binding energy of the new atom [eV]
    @return output: dictionary with the scaled cross section.
    """

    if 'atom' not in xsection:
        raise Exception('The input information must contain the atomic data')

    if 'base' not in xsection:
        raise Exception('The input cross-section must have an energy base')

    if 'sigma' not in xsection:
        raise Exception('The cross-section is empty')


    # The energy where the cross-section is evaluated must be first scaled
    # with the ratio between the original binding energy and the new binding
    # binding energy.
    ratU = xsection['atom']['binding_energy']/Ubnew

    if xsection['atom']['binding_unit'] == 'J':
        ratU = ratU * sspar.ec

    if xsection['base_units'] == 'J/kg':
        xsection['base_units'] = 'eV/amu'
        xsection['base'] *= sspar.ec*sspar.kg2amu
    elif xsection['base_units'] == 'J':
        xsection['base_units'] = 'eV'
        xsection['base'] *= sspar.ec

    output = {}
    output['base']  = ratU * xsection['base']
    output['base_units'] = 'eV'
    output['base_type'] = 'e'
    output['sigma'] = xsection['sigma'] * ratU**2.0
    output['units'] = xsection['units']
    output['name_long'] = None
    output['name_short'] = None
    output['atom'] = {
                        'name': None,
                        'symbol': None,
                        'binding_energy': Ubnew,
                        'unit': 'eV'
                     }
    return output

# ----------------------------------------------------------
#--- Caesium and Rubidium (neutrals-GS) cross-sections
# ----------------------------------------------------------
def Lotz_cross_section(Ebase: float, chi: float, occNum: int,
                       atom: dict = None):
    """
    Electron-impact ionization cross section according to:
    Astrophysical Journal Supplement, vol. 14, p.207 (1967)
    http://adsabs.harvard.edu/doi/10.1086/190154

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ebase: array of energy values to compute the cross-section [eV]
    @param chi: list of ionization energies [eV]
    @param occNum: list with the occupation numbers of the levels to excite.
    There must be the same number as in len(chi)
    @param atom: dictionary with the atomic data (name, symbol, binding
    energy,...)

    @return output: a dictionary containing the cross section evaluated in
    the input points, along with further info

    """

    #--- Checking the inputs
    if len(chi) != len(occNum):
        raise Exception('Ionization potential and occupation numbers \
                        must have the same length')

    if atom is None:
        atom = { 'name': 'Alkali',
                 'symbol': 'Ns^1',
                 'binding_energy': chi[0],
                 'binding_units': 'eV'

               }

    #--- Generating the output dictionary.
    output = {
                'base': Ebase,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': 'E',
                'base_type': 'e',
                'atom': atom,
                'sigma': np.zeros(len(Ebase)),
                'units': '$cm^2$'
             }

    #--- Loop over all the ionization potentials.
    for ii in range(len(chi)):
        aux = occNum[ii]*np.log(Ebase/chi[ii])/(chi[ii]*Ebase)
        aux[np.where(aux <= 0.0)] = 0.0
        output['sigma'] += aux

    output['sigma'] *= 1.0e-14


    return output

def Cs0_CX_H_v1(Ecm: float, redMass: float = reducedMasses['Cs_D']):
    """
    Charge-exchange reaction between Cs0 and H+:
        Cs0 + D+  => Cs+ + D0

    F.W. MEYER and L.W. ANDERSON, Phys. Lett. 54 A, Vol 4 (1975)

    Cs0 + D+ => Cs+ + D0

    The cross section depends on the relative velocity between the exchange
    partners (see fig. 1 in reference)

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ecm: Energy in the center of mass.
    @return output: dictionary with the cross section evaluated.
    """

    # Values read from Fig. 1
    v = np.array([0.5e6, 0.75e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6])
    sig = np.array([1.4e-14, 1e-14, 0.65e-14, 0.2e-14, 0.12e-14, 0.1e-14])

    # Create the interpolating object.
    sig_interp = interp1d(np.log(v), np.log(sig), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)
    vrel = np.sqrt(2.0*Ecm*sspar.ec/(redMass*sspar.amu2kg))
    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'E',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(vrel))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': Cs0_6s1,
                         'binding_units': 'eV'
                       },
                'reaction': 'Cs^0 + H^+ => Cs^+ + H^0',
                'reaction_name': 'Cs charge exchange with H'
             }
    return output

def Cs0_CX_H_v2(Ecm: float, massBackg: float = alkMasses['D']):
    """
    Cs Charge exchange transfer cross section
    according to

    FW Meyer, J. Phys. B 13, 3823 (1980).

    http://iopscience.iop.org/article/10.1088/0022-3700/13/19/020

    It is newer, but for a less broad energy range

    The cross section depends on the relative velocity between the exchange
    partners (see fig. 1 in reference)

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ecm: center-of-mass energy [eV]
    @param massH: AMU of the hydrogen partner in the collision.
    @return output: dictionary with the cross section data.
    """

    # From paper:
    EperAMU = np.array([100., 150., 225., 300., 350., 400., 500., 600., 800.,
                        1000., 1500., 2000., 4000.])

    E = EperAMU*massBackg

    # Cross section in cm^2
    sig = np.array([1.06, 1.47, 1.47, 1.64, 1.61, 1.77, 1.74, 1.79, 1.56, 1.35,
                    1.27, 1.18, 0.8])*1.0e-14

    # Interpolating object
    sig_interp = interp1d(np.log(E), np.log(sig), kind='linear',
                           fill_value='extrapolate', assume_sorted=True,
                           bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': Cs0_6s1,
                         'binding_units': 'eV'
                       },
                'reaction': 'Cs^0 + H^+ => Cs^+ + H^0',
                'reaction_name': 'Cs charge exchange with H'
             }
    return output

def Cs0_CX_H(Ecm: float, isotopeH: str = 'D'):
    """
    This will combine the two Cs CX cross sections.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ecm: energy in the center of mass to evaluate the cross sections.
    In eV.
    @param isotopeH: type of hydrogen isotope to make the calculations ('H',
    'D', 'T')
    @return output: dictionary with the cross section data.
    """

    redMass = reducedMasses['Cs_'+isotopeH]
    massH = alkMasses[isotopeH]

    cs_v1 = Cs0_CX_H_v1(Ecm, redMass)
    cs_v2 = Cs0_CX_H_v2(Ecm, massH)

    ratu2 = (H0_1s1/Cs0_6s1)**2
    Ecm_mod_H = Ecm/ratu2
    cs_v3 = Wutte_scaling(H0_proton_CX(Ecm_mod_H), Ubnew=Cs0_6s1)

    ratu2 = (Li0_2s1/Cs0_6s1)**2
    Ecm_mod_H = Ecm/ratu2
    cs_v4 = Wutte_scaling(Li_proton_CX_ion(Ecm_mod_H), Ubnew=Cs0_6s1)
    cs_out = cs_v1
    cs_out['sigma'] = np.maximum(cs_v1['sigma'],
                                 np.maximum(cs_v2['sigma'],
                                 np.maximum(cs_v3['sigma'], cs_v4['sigma'])))

    return cs_out

def Rb0_CX_Ebel(Ecm, massBackg: float = alkMasses['D']):
    """
    Charge exchange transfer cross section D with Rubidium
    according to

    F Ebel and E Salzborn

    Journal of Physics B: Atomic and Molecular Physics, Volume 20, Number 17
    http://iopscience.iop.org/article/10.1088/0022-3700/20/17/029

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ecm: center-of-momentum energy [eV]
    @param massBackg: mass of the hydrogen atom with which it is colliding.
    @return output: dictionary with the cross section data.
    """

    # In the table, what appears is the energy per AMU.
    EperAMU = np.array([200., 300., 400., 500., 600., 700., 800., 1000.,
                        1200., 1500., 2000.,  3500.,4000., 5000.])
    E = EperAMU * massBackg

    # Cross-sections in the table:
    sig = np.array([6.5, 8.0, 8.3, 8.8, 9.7, 10.5, 9.5, 9.7, 10., 9.0, 7.4,
                    6.8, 6.5, 5.8])*1.0e-15 # cm^2

    # Interpolating object
    sig_interp = interp1d(np.log(E), np.log(sig), kind='quadratic',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Rubidium',
                         'symbol': 'Rb',
                         'binding_energy': Rb0_5s1,
                         'binding_units': 'eV'
                       },
                'reaction': 'Rb^0 + H^+ => Rb^+ + H^0',
                'reaction_name': 'Rb charge exchange with H'
             }
    return output

def Rb0_CX_Girnius(Ecm, massBackg: float = alkMasses['D']):
    """
    Charge exchange transfer cross section D with Rubidium
    according to

    R.J.Girnius, L.W.Anderson, E.Staab
    Nuclear Instruments and Methods
    Volume 143, Issue 3, 15 June 1977, Pages 505-511

    https://www.sciencedirect.com/science/article/pii/0029554X77902397

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ecm: center-of-momentum energy [eV]
    @param massBackg: mass of the hydrogen atom with which it is colliding.
    @return output: dictionary with the cross section data.
    """

    # In the table, what appears is the energy per AMU.
    EperAMU = np.array([1.1, 1.5, 2.0, 3.0, 4.0, 5.0,
                        6.0, 10.0, 15.0, 20.0])*1000.
    E = EperAMU * massBackg

    # Cross-sections in the table:
    sig = np.array([1039.0, 1140.0, 1080.0, 1065.0, 1010.0, 980.0, 930.0,
                   653.0, 450.0, 260.0])*1.0e-17

    # Interpolating object
    sig_interp = interp1d(np.log(E), np.log(sig), kind='quadratic',
                           fill_value='extrapolate', assume_sorted=True,
                           bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Rubidium',
                         'symbol': 'Rb',
                         'binding_energy': Rb0_5s1,
                         'binding_units': 'eV'
                       },
                'reaction': 'Rb^0 + H^+ => Rb^+ + H^0',
                'reaction_name': 'Rb charge exchange with H'
             }
    return output

def Rb0_CX_H(Ecm: float, isotopeH: str = 'D'):
    """
    This will combine the two Rb CX cross sections, by considering the worst-
    case scenario.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param Ecm: energy in the center of mass to evaluate the cross sections.
    In eV.
    @param isotopeH: type of hydrogen isotope to make the calculations ('H',
    'D', 'T')
    @return out: dictionary with the cross section.
    """

    massH = alkMasses[isotopeH]

    rb_v1 = Rb0_CX_Ebel(Ecm, massH)
    rb_v2 = Rb0_CX_Girnius(Ecm, massH)

    out = rb_v1
    out['sigma'] = np.maximum(rb_v1['sigma'], rb_v2['sigma'])

    return out

# ----------------------------------------------------------
#--- Caesium and Rubidium (ionized-GS) cross-sections
# ----------------------------------------------------------
def CsI_CsII_electron(Ecm: float):
    """
    Electron impact ionization from the CsI => CsII

    Cs+ + e => Cs++ + 2e

    Data from:

    Data Number 11

    Cs+ + e --> Cs2+ + 2e
    Hertling, D.R. et al.
    J. Appl. Phys. 53 (1982) 5427
    NDP = 26

    @param Ecm: energy in the center of mass (eV)
    @return output: dictionary with the cross section data.
    """

    E = np.array((2.800000e+01, 	3.300000e+01, 3.800000e+01, 	4.800000e+01,
                  5.800000e+01, 6.800000e+01, 7.800000e+01, 	8.800000e+01,
                  9.800000e+01, 	1.180000e+02, 1.380000e+02, 	1.680000e+02,
                  1.980000e+02, 	2.180000e+02, 	2.480000e+02, 2.980000e+02,
                  3.980000e+02, 	4.980000e+02, 	5.980000e+02, 7.480000e+02,
                  9.980000e+02, 	1.498000e+03, 	1.998000e+03, 	2.998000e+03,
                  3.998000e+03, 4.998000e+03)) # [eV]
    sig = np.array((1.210000e-16, 	1.670000e-16,   1.970000e-16,
                    2.090000e-16,   2.050000e-16,   2.090000e-16,
                    2.110000e-16, 	2.110000e-16,	2.090000e-16,
                    1.910000e-16,   1.720000e-16, 	1.610000e-16,
                    1.530000e-16,	1.370000e-16,	1.270000e-16,
                    1.190000e-16, 	9.390000e-17, 	8.200000e-17,
                    7.160000e-17,	6.320000e-17,   5.150000e-17,
                    3.640000e-17, 	2.900000e-17, 	2.000000e-17,
                    1.540000e-17,   1.310000e-17))# [cm^2]

    sig_interp = interp1d(np.log(E), np.log(sig), kind='linear',
                           fill_value='extrapolate', assume_sorted=True,
                           bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': CsI_5p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'CsI + e => CsII',
                'reaction_name': 'Cs+ ionization'
             }
    return output

def CsI_CsIII_electron(Ecm: float):
    """
    Electron impact ionization from the CsI => CsIII

    Cs+ + e => Cs+++ + 3e

    Data Number 11

    Cs+ + e --> Cs3+ + 3e
    Hertling, D.R. et al.
    J. Appl. Phys. 53 (1982) 5427
    NDP = 26

    @param Ecm: energy in the center of mass (eV)
    @return output: dictionary with the cross section data.
    """

    E = np.array((6.800000e+01, 	7.800000e+01, 	8.800000e+01, 	9.800000e+01,
                  1.180000e+02, 1.380000e+02, 	1.680000e+02, 	1.980000e+02,
                  2.180000e+02, 	2.480000e+02,   2.980000e+02, 	3.980000e+02,
                  4.980000e+02, 	5.980000e+02, 	7.480000e+02,   9.980000e+02,
                  1.498000e+03, 	1.998000e+03, 	2.998000e+03, 	3.998000e+03,
                  4.998000e+03)) # [eV]
    sig = np.array([3.980000e-18, 	7.170000e-18, 	1.320000e-17,
                    2.460000e-17, 	3.740000e-17,   3.570000e-17,
                    3.030000e-17, 	2.760000e-17, 	2.560000e-17,
                    2.440000e-17,   2.500000e-17, 	2.360000e-17,
                    2.250000e-17, 	2.070000e-17, 	1.930000e-17,
                    1.720000e-17, 	1.400000e-17, 	1.190000e-17,
                    8.910000e-18, 	7.190000e-18,   6.130000e-18]) # [cm^2]

    sig_interp = interp1d(np.log(E), np.log(sig), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': CsI_5p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'CsI + e => CsIII',
                'reaction_name': 'Cs+ ionization to III'
             }
    return output

def CsI_CsIV_electron(Ecm: float):
    """
    Electron impact ionization from the CsI => CsIV

    Cs+ + e => Cs4+ + 3e

    Data from:

    Data Number 16

    Cs+ + e --> Cs4+ + 4e
    Hertling, D.R. et al.
    J. Appl. Phys. 53 (1982) 5427
    NDP = 17

    @param Ecm: energy in the center of mass (eV)
    @return output: dictionary with the cross section data.
    """
    E = np.array((  1.180000e+02, 	1.380000e+02, 	1.680000e+02,
                    1.980000e+02, 	2.180000e+02,   2.480000e+02,
                    2.980000e+02, 	3.980000e+02, 	4.980000e+02,
                    5.980000e+02,   7.480000e+02, 	9.980000e+02,
                    	1.498000e+03, 	1.998000e+03, 	2.998000e+03,
                    3.998000e+03, 	4.998000e+03)) # [eV]
    sig = np.array([1.060000e-18, 	3.850000e-18, 	5.700000e-18,
                    7.560000e-18, 	8.260000e-18,   7.570000e-18,
                    7.210000e-18, 	6.530000e-18, 	5.930000e-18,
                    5.310000e-18,   4.460000e-18, 	3.620000e-18,
                    2.700000e-18, 	2.210000e-18, 	1.530000e-18,
                    1.170000e-18, 	9.260000e-19])  # [cm^2]

    sig_interp = interp1d(np.log(E), np.log(sig), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': CsI_5p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'CsI + e => CsIV',
                'reaction_name': 'Cs+ ionization to IV'
             }
    return output

def CsI_CsV_electron(Ecm: float):
    """
    Electron impact ionization from the CsI => CsV

    Cs+ + e => Cs5+ + 4e

    Data from:

    Data Number 17

    Cs+ + e --> Cs5+ + 5e
    Hertling, D.R. et al.
    J. Appl. Phys. 53 (1982) 5427
    NDP = 14

    @param Ecm: energy in the center of mass (eV)
    @return output: dictionary with the cross section data.
    """
    E = np.array((  1.980000e+02, 	2.180000e+02, 	2.480000e+02,
                    2.980000e+02, 	3.980000e+02,   4.980000e+02,
                    5.980000e+02, 	7.480000e+02, 	9.980000e+02,
                    1.498000e+03,   1.998000e+03, 	2.998000e+03,
                    3.998000e+03, 	4.998000e+03)) # [eV]
    sig = np.array([3.480000e-19, 	9.230000e-19, 	1.520000e-18,
                    2.100000e-18, 	2.060000e-18,   2.030000e-18,
                    1.940000e-18, 	1.770000e-18, 	1.500000e-18,
                    1.250000e-18,   1.050000e-18, 	7.830000e-19,
                    5.920000e-19, 	4.990000e-19]) # [cm^2]

    sig_interp = interp1d(np.log(E), np.log(sig), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': CsI_5p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'CsI + e => CsV',
                'reaction_name': 'Cs+ ionization to V'
             }
    return output

def CsI_CsN_electron(Ecm: float):
    """
    Combines the cross-section data from all the CsI->Cs2+,Cs3+,...

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param: Center-of-mass energy of the Cs beam.
    @return output: dictionary with the cross section data.
    """

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.zeros(len(Ecm)),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Caesium',
                         'symbol': 'Cs',
                         'binding_energy': CsI_5p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'CsI + e => CsN+',
                'reaction_name': 'Secondary ionization of Cs+'
             }

    output['sigma'] += CsI_CsII_electron(Ecm)['sigma']
    output['sigma'] += CsI_CsIII_electron(Ecm)['sigma']
    output['sigma'] += CsI_CsIV_electron(Ecm)['sigma']
    output['sigma'] += CsI_CsV_electron(Ecm)['sigma']
    return output

def RbI_RbII_electron(Ecm: float):
    """
    Electron impact ionization for RbI to RbII.

    Data from:
    M.J. Higgins, M.A. Lennon, J.G. Hughes, K.L. Bell, H.B. Gilbody,
    A.E. Kingston, F.J. Smith CLM-R294 (1989)

    @param Ecm: center-of-mass energy of the impact [eV]
    @return output: dictionary with the cross section data.
    """

    E = np.array((  2.7300E+01, 3.1234E+01, 3.5735E+01, 4.0885E+01, 4.6777E+01,
                    5.3517E+01, 6.1230E+01, 7.0053E+01, 8.0149E+01, 9.1699E+01,
                    1.0491E+02, 1.2003E+02, 1.3733E+02, 1.5712E+02, 1.7976E+02,
                    2.0567E+02, 2.3530E+02, 2.6921E+02, 3.0801E+02, 3.5239E+02,
                    4.0318E+02, 4.6128E+02, 5.2775E+02, 6.0380E+02, 6.9082E+02,
                    7.9037E+02, 9.0427E+02, 1.0346E+03, 1.1837E+03, 1.3542E+03,
                    1.5494E+03, 1.7727E+03, 2.0281E+03, 2.3204E+03, 2.6548E+03,
                    3.0374E+03, 3.4751E+03, 3.9758E+03, 4.5488E+03, 5.2043E+03,
                    5.9543E+03, 6.8123E+03, 7.7941E+03, 8.9172E+03, 1.0202E+04,
                    1.1672E+04, 1.3355E+04, 1.5279E+04, 1.7481E+04,
                    2.0000E+04))

    sig = np.array([ 0.0000E+00, 3.9313E-17, 7.5178E-17, 1.0559E-16,
                     1.2965E-16, 1.4729E-16, 1.5896E-16, 1.6539E-16,
                     1.6743E-16, 1.6596E-16, 1.6178E-16, 1.5559E-16,
                     1.4800E-16, 1.3950E-16, 1.3050E-16, 1.2130E-16,
                     1.1214E-16, 1.0319E-16, 9.4585E-17, 8.6399E-17,
                     7.8688E-17, 7.1481E-17, 6.4787E-17, 5.8603E-17,
                     5.2916E-17, 4.7707E-17, 4.2950E-17, 3.8620E-17,
                     3.4687E-17, 3.1123E-17, 2.7899E-17, 2.4989E-17,
                     2.2365E-17, 2.0002E-17, 1.7877E-17, 1.5968E-17,
                     1.4255E-17, 1.2719E-17, 1.1343E-17, 1.0111E-17,
                     9.0094E-18, 8.0242E-18, 7.1440E-18, 6.3580E-18,
                     5.6565E-18, 5.0307E-18, 4.4727E-18, 3.9754E-18,
                     3.5323E-18, 3.1378E-18])

    sig_interp = interp1d(np.log(E), np.log(sig), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Rubidium',
                         'symbol': 'Rb',
                         'binding_energy': RbI_4p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'RbI + e => RbII',
                'reaction_name': 'Rb+ secondary ionization'
             }
    return output

def RbI_RbIII_electron(Ecm: float):
    """
    Electron impact ionization from RbI to RbIII

    Data from:
    Hughes D.W. & Feeney R. K.   Physics Review A23 2241  (1981)
    "Absolute experimental cross sections for the electron-impact multiple
    ionization of singly charged rubidium ions"

    @param Ecm: center-of-mass energy of the impact [eV]
    @return output: dictionary with the cross section data.
    """
    E = [98.0, 118.0, 138.0, 168.0, 188.0, 248.0, 298.0, 398.0, 498.0, 598.0,
         698.0, 798.0, 898.0, 998.0, 1248.0, 1498.0, 1998.0, 2998.0]

    sigma = [0.00, 0.00, 1.00, 2.60 , 5.27, 10.1, 13.3, 17.2, 20.0, 21.4,
             20.9, 20.6, 20.3, 19.7, 18.5, 17.1, 15.7, 11.9]*1.0e-19 # cm^2

    sig_interp = interp1d(np.log(E), np.log(sigma), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Rubidium',
                         'symbol': 'Rb',
                         'binding_energy': RbI_4p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'RbI + e => RbIII',
                'reaction_name': 'Rb+ secondary ionization'
             }
    return output

def RbI_RbIV_electron(Ecm: float):
    """
    Electron impact ionization from RbI to RbIV

    Data from:
    Hughes D.W. & Feeney R. K.   Physics Review A23 2241  (1981)
    "Absolute experimental cross sections for the electron-impact multiple
    ionization of singly charged rubidium ions"

    @param Ecm: center-of-mass energy of the impact [eV]
    @return output: dictionary with the cross section data.
    """
    E = [158.0, 198.0, 248.0, 298.0, 398.0, 498.0, 598.0, 698.0, 798.0, 898.0,
         998.0, 1248.0, 1498.0, 1998.0, 2998.0]

    sigma = [0.00, 0.00, 3.50, 7.30, 24.1, 36.9, 41.8, 47.3, 51.1, 50.1,
             49.8, 46.3, 43.2, 38.8, 27.6]*1.0e-20 # cm^2

    sig_interp = interp1d(np.log(E), np.log(sigma), kind='linear',
                          fill_value='extrapolate', assume_sorted=True,
                          bounds_error=False)

    output = {
                'base': Ecm,
                'base_units': 'eV',
                'base_name': 'Energy',
                'base_short': '$E_{CM}$',
                'base_type': 'e',
                'sigma': np.exp(sig_interp(np.log(Ecm))),
                'units': '$cm^2$',
                'atom':  {
                         'name': 'Rubidium',
                         'symbol': 'Rb',
                         'binding_energy': RbI_4p6,
                         'binding_units': 'eV'
                       },
                'reaction': 'RbI + e => RbIII',
                'reaction_name': 'Rb+ secondary ionization'
             }
    return output

def RbI_RbN_electron(Ecm: float):
    """
    Combines the cross-section data from all the RbI->Rb2+,Rb3+,...

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param: Center-of-mass energy of the Rb beam.
    @return output: dictionary with the cross section data.
    """

    output = {
        'base': Ecm,
        'base_units': 'eV',
        'base_name': 'Energy',
        'base_short': '$E_{CM}$',
        'base_type': 'e',
        'sigma': np.zeros(len(Ecm)),
        'units': '$cm^2$',
        'atom':  {
                  'name': 'Rubidium',
                  'symbol': 'Rb',
                  'binding_energy': RbI_4p6,
                  'binding_units': 'eV'
                },
        'reaction': 'RbI + e => RbN+',
        'reaction_name': 'Secondary ionization of Rb+'
    }

    output['sigma'] += RbI_RbII_electron(Ecm)['sigma']
    output['sigma'] += RbI_RbIII_electron(Ecm)['sigma']
    output['sigma'] += RbI_RbIV_electron(Ecm)['sigma']
    return output
# ----------------------------------------------------------
#--- Calculation of the reactivity
# ----------------------------------------------------------
"""
In this section, all the routines to compute the reactivities,
provided the cross-sections as a function of the energy. Maxwellian function
is typically used for the calculation.
"""

def maxwell(v, T: float, mass: float):
    """
    Maxwell-Boltzmann distribution function.

    @param v: velocity array to evaluate the distribution.
    @param T: temperature scalar given in eV.
    @param mass: mass of the species.

    """

    beta = np.sqrt(mass/(2.0*T*sspar.ec))
    fmb = (beta/np.sqrt(np.pi))**3.0 * np.exp(-(beta**2) * (v**2))

    return fmb

def maxwell_jutner(v, T: float, mass:float):
    """
    Maxwell-Boltzmann-Juetner distribution function.
    Relativistic ideal gas distribution function.

    @param v: velocity array to evaluate the distribution.
    @param T: temperature scalar given in eV.
    @param mass: mass of the species.
    """

    # Computing the relativistic factor
    beta_rel = v/sspar.c
    gamma_rel = 1.0/np.sqrt(1.0-beta_rel**2)

    # Thermal to mass ratio.
    theta = T*sspar.ec/(mass*sspar.c**2)
    theta1 = 1.0/theta

    # Evaluating the modified Bessel function term:
    k2 = modBessel2nd(2, theta1)

    # MBJ distribution:
    f = gamma_rel**2*beta_rel/(theta*k2)*np.exp(-gamma_rel*theta1)

    return f

def reactionRate(xsection: dict, massA: float, massB: float,
                 T: float = None, vel: float = None):
    """
    Computes the Maxwellian reaction rate, for the given cross-section.
    Reaction-rate is computed for an input array of temperatures and given the
    masses of the reactants in AMU.
        A + B --> C
    If one of the masses is of the order of the electron mass whilst the other
    is an ionic mass, the routine will approximate the relative velocity by
    the ion velocity. Otherwise, the cross-section will depend both on the
    temperature and the velocity.


    @param xsection: dictionary with all the cross-section data.
    @param T: temperature array where the reaction-rate is going to be
    computed. If it is not provided, an array linearly spaced between 1 eV and
    10 keV is created.
    @param vel: velocity array where the reaction rate is calculated. If
    the masses of the system as quite different (i.e., electron colliding with
    a proton), this will be disregarded. If it is not provided, the velocity
    basis will be the velocities equivalent to the cross-section basis.
    @param massA: mass in AMU of the main reactant.
    @param massB: mass in AMU of the secondary reactant.

    @return
    """

    #--- Checking the inputs.
    if 'sigma' not in xsection:
        raise Exception('No cross-section data found!')
    if 'base' not in xsection:
        raise Exception('The energy where the cross-section is defined \
                         is not in the input input!')
    if 'atom' not in xsection:
        raise Exception('Atom data must be provided.')

    ignore_velrel = False
    quotient = massA/massB

    if quotient > 1500.0:
        ignore_velrel = True

    #--- Transforming the base values (energy) into velocities.
    if xsection['base_type'] == 'v':
        velocity = xsection['base']
    elif xsection['base_units'] == 'eV':
        velocity = np.sqrt(2.0*xsection['base']*sspar.ec/(massA*sspar.amu2kg))
    elif xsection['base_units'] == 'keV':
        velocity = np.sqrt(2.0*xsection['base']*sspar.ec*1.0e3/ \
                           (massA*sspar.amu2kg))
    elif xsection['base_units'] == 'eV/amu':
        velocity = np.sqrt(2.0*xsection['base']*sspar.ec/sspar.amu2kg)
    elif xsection['base_units'] == 'keV/amu':
        velocity = np.sqrt(2.0*xsection['base']*sspar.ec*1.0e3/sspar.amu2kg)
    else:
        raise Exception('The input units for the base cross-sections \
                         are not valid')
    if vel is None and not ignore_velrel:
        vel = velocity

    if T is None:
        T = np.logspace(0, 5.)

    #--- Creating the variables to store.
    output = {}
    output['base'] = []
    output['base'].append({
                            'units': 'eV',
                            'name': 'Temperature',
                            'short': 'T',
                            'data': T,
                            'useful': True
                          })
    # The cross-section has to evaluated in the center-of-mass frame:
    massB_si = massB * sspar.amu2kg
    redMass = massA * massB / (massA + massB)*sspar.amu2kg

    if not ignore_velrel:
        output['base'].append({
                            'units': 'm/s',
                            'name': 'Velocity',
                            'short': '$V_{beam}$',
                            'data': vel,
                            'useful': True
                          })
        output['base'].append({
                            'units': 'keV',
                            'name': 'Energy',
                            'short': 'E',
                            'data': 0.5*redMass*vel**2/sspar.ec*1.0e-3,
                            'useful': False
                          })


        output['rate'] = np.zeros((len(T), len(vel)))
        output['drate'] = np.zeros((len(T), len(vel)))
        for iT in tqdm(range(len(T))):
            # Callable function for Maxwell distribution.
            fM = lambda vx,vy,vz: maxwell(np.sqrt(vx**2 + vy**2 + vz**2),
                                          T[iT], massB_si)

            # The integration is typically limited by the numerical
            # accuracy. If really small number are provided, it will fail.
            # Hence, all the normalizations are taken away:

            #norm_maxwell=np.sqrt(massB_si/(2.0*T[iT]*sspar.ec))**3.0
            # To perform the numerical integration, we will use the
            # maximum velocity:
            vmax =  np.sqrt((2.0*T[iT]*sspar.ec)/(massB_si))*3.0

            for iv in range(len(vel)):

                # Center-of-mass energy:
                vrel  = lambda vx,vy,vz: np.sqrt((vx-vel[iv])**2
                                                 + vy**2
                                                 + vz**2)
                Ecm   = lambda vx,vy,vz: 0.50*redMass*\
                                         ((vx-vel[iv])**2 +vy **2
                                         + vz**2)/sspar.ec
                # Spline interpolator for the cross-section.
                sigma_aux = interp1d(np.log(xsection['base']),
                                     np.log(xsection['sigma']),
                                     fill_value=-np.inf,
                                     kind='linear',
                                     assume_sorted=True,
                                     bounds_error=False)

                if xsection['base_type'] == 'v':
                    sigmaCM = lambda vx, vy, vz: \
                              np.exp(sigma_aux(np.log(vrel(vx,vy,vz))))*1.0e-4
                else:
                    sigmaCM = lambda vx, vy, vz: \
                              np.exp(sigma_aux(np.log(Ecm(vx,vy,vz))))*1.0e-4

                # The integrand is:
                func = lambda vx,vy,vz: vrel(vx, vy, vz)*\
                                        fM(  vx, vy, vz)*\
                                        sigmaCM(vx, vy, vz)

                # We perform a triple integral.
                aux = tplquad(func, -vmax, vmax,
                              lambda x: -vmax, lambda x:  vmax,
                              lambda x, y: -vmax, lambda x,y: vmax)
                output['rate'][iT, iv] =  aux[0]
                output['drate'][iT, iv] = aux[1]


    else:
        output['rate'] = np.ones(len(T))
        for iT in tqdm(range(len(T))):
            # Spline interpolator for the cross-section.
            sigma_aux = interp1d(np.log(xsection['base']),
                                 xsection['sigma'],
                                 fill_value='extrapolate',
                                 kind='linear',
                                 assume_sorted=True,
                                 bounds_error=False)
            if xsection['base_type'] == 'v':
                sigmaCM = lambda v: \
                          sigma_aux(np.log(v))

            else:
                sigmaCM = lambda v:  \
                          sigma_aux(np.log(0.50*v**2*redMass/sspar.ec))

            # If the temperature is comparable to the rest mass energy,
            # the Maxwellian distribution must be replaced by its relativistic
            # counterpart.
            if T[iT]/(massB_si*sspar.c**2.0) > 0.01:
                fmaxwell = lambda v: maxwell_jutner(v, T[iT], massB_si)
            else:
                fmaxwell = lambda v: maxwell(v, T[iT], massB_si)


            # We perform the 1D integral.
            # The integrand is:
            func = lambda v: v**3.0*sigmaCM(v)*fmaxwell(v)
            output['rate'][iT] = quad(func, 0., 3e8)[0]*4.0*np.pi*1.0e-4

    output['units'] = '$m^{-3}s^{-1}$'
    output['atom'] = xsection['atom']

    return output

# ----------------------------------------------------------
#--- Database generators
# ----------------------------------------------------------
"""
This section will use the data above to generate a self-consistent reaction
rate database for an alkali atom.
"""

def create_HIBP_tables():
    """
    This routines creates all the ionization rates tables for the Rb and Cs
    species (the used in the i-HIBP diagnostic) and store them into mat files.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    #cs = createAlkaliData('Cs', 0)
    #savemat("Cs133_tables.mat", cs)

    rb85 = createAlkaliData('Rb', 85)
    savemat("Rb85_tables.mat", rb85)

    return rb85


def createAlkaliData(name: str, mass: float, T: float = None,
                     vel: float = None, plotflag: bool = True,
                     gridflag: bool = True, fontname = 'LM Roman 10',
                     fontsize = 11, line_options: dict = None,
                     impurities: bool = True):
    """
    This computes and stores to a dictionary the reaction-rates of all the
    possible interaction of an alkali beam and a plasma.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param name: name of the Alkali element to make the table for. Up to now,
    only 'Cs' and 'Rb' are allowed.
    @param mass: when there are more than one stable alkali element
    (i.e., 85Rb, 87Rb), this value should provide the mass number (85, 87 in
    the previous example).
    @param T: temperatures to evaluate the reaction rates. If not provided,
    a logarithmically spaced array is set by default between 10 eV and 100 keV
    @param vel: velocities to evaluate the reaction rates. If not provided,
    a linearly spaced array is set by default between 9e5 - 7e6 m/s.
    @param plotflag: Flag to plot the reaction rates. True by default.
    @param gridflag: Flag to plot the grids in the plots. True by default.
    @param fontname: fontname to use in the plots.
    @param fontsize: fontsize for the plots.
    @para line_options: dictionary for the lines plotting.
    @returns output: returns the dictionary with the different reactions.
    """
    #--- Checking the inputs.
    if T is None:
        T = np.logspace(1.0, 5.0, num=64)

    if vel is None:
        vel = np.linspace(1.0e6, 6.0e6, num=10)

    #--- Chcking the line options input:
    if line_options is None:
        line_options = dict()
        line_options['linewidth'] = 2
    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2

    if name == 'Cs':
        return createCsData(T, vel, plotflag=plotflag, gridflag=gridflag,
                            fontname=fontname, fontsize=fontsize,
                            line_options=line_options,
                            impurities=impurities)

    if name == 'Rb':
       return createRbData(mass, T, vel, plotflag=plotflag, gridflag=gridflag,
                           fontname=fontname, fontsize=fontsize,
                           line_options=line_options,
                           impurities=impurities)
    return 0

def createCsData(T: float, vel: float, plotflag: bool = True,
                 gridflag: bool = True, fontname = 'LM Roman 10',
                 fontsize = 11, line_options: dict = None,
                 impurities: bool = False):
    """
    Computes the reaction rates of the 133Cs.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param T: temperatures to evaluate the reaction rates.
    @param vel: velocities to evaluate the reaction rates.
    @param plotflag: Flag to plot the reaction rates. True by default.
    @param gridflag: Flag to plot the grids in the plots. True by default.
    @param fontname: fontname to use in the plots.
    @param fontsize: fontsize for the plots.
    @para line_options: dictionary for the lines plotting.
    @returns output: returns the dictionary with the different reactions.
    """
    mass = alkMasses['Cs133']
    E = np.logspace(0, 7, num=512)
    # Ionization potentials of Cs
    chi = np.array((Cs0_6s1, CsI_5p6))
    occNum = np.array((1, 6))

    #--- Electron impact ionization.
    cs0_csI_electron_data    = Lotz_cross_section(E, chi, occNum)
    cs0_csI_CX_data          = Cs0_CX_H(E)
    csI_csII_electron_data   = CsI_CsII_electron(E)
    csI_csIII_electron_data  = CsI_CsIII_electron(E)
    csI_csIV_electron_data   = CsI_CsIV_electron(E)
    csI_csV_electron_data    = CsI_CsV_electron(E)
    csI_csN_electron_data    = CsI_CsN_electron(E)

    #--- Ion impact ionization.
    # We approximate the ion-impact ionization from the H formulae
    cs0_csI_ion_data        = Wutte_scaling(H0_proton_imp_ion(E),
                                            Ubnew=Cs0_6s1)

    if impurities:
        #--- Impurity impact ionization.

        cs0_csI_B5_Li_data      = Li_heavy_imp_ion(E, +5, Ub=Cs0_6s1)

        # For the impurity ionization we choose the worst-case scenario:
        cs0_csI_B5_data = cs0_csI_B5_Li_data


        cs0_csI_C6_Li_data     = Li_heavy_imp_ion(E, +6, Ub=Cs0_6s1)

        # For the impurity ionization we choose the worst-case scenario:
        cs0_csI_C6_data = cs0_csI_C6_Li_data

        #--- Impurity CX ionization.
        # We approximate the B+5-CX ionization from the H formulae
        cs0_csI_B5_H_data_CX    = Wutte_scaling(H0_B5_CX(E),
                                                Ubnew=Cs0_6s1)

        # We approximate the C+6-impact ionization from the H formulae
        cs0_csI_C6_H_data_CX    = Wutte_scaling(H0_C6_CX(E),
                                               Ubnew=Cs0_6s1)

    #--- Computing the reaction-rates:
    cs0_csI_electron_rate = reactionRate(cs0_csI_electron_data,
                                         massA = mass,
                                         massB = sspar.mass_electron_amu,
                                         T=T, vel=vel)

    csI_csN_electron_rate = reactionRate(csI_csN_electron_data,
                                   massA = mass,
                                   massB = sspar.mass_electron_amu,
                                   T=T, vel=vel)

    cs0_csI_ion_rate = reactionRate(cs0_csI_ion_data,
                                    massA = mass, massB = alkMasses['D'],
                                    T=T, vel=vel)

    cs0_csI_CX_rate = reactionRate(cs0_csI_CX_data,
                                    massA = mass, massB = alkMasses['D'],
                                    T=T, vel=vel)
    if impurities:
        cs0_csI_C6_rate = reactionRate(cs0_csI_C6_data,
                                       massA = mass, massB = alkMasses['C'],
                                       T=T, vel=vel)

        cs0_csI_B5_rate = reactionRate(cs0_csI_B5_data,
                                       massA = mass, massB = alkMasses['B'],
                                       T=T, vel=vel)

        cs0_csI_C6_CX_rate = reactionRate(cs0_csI_C6_H_data_CX,
                                          massA = mass, massB = alkMasses['C'],
                                          T=T, vel=vel)

        cs0_csI_B5_CX_rate = reactionRate(cs0_csI_B5_H_data_CX,
                                          massA = mass, massB = alkMasses['B'],
                                          T=T, vel=vel)
    #--- Creating the output variable:
    output = { 'date': str(date.today()),
        'species': 'Cs133',
        'species_symbol': '${}^{133}Cs$',
        'rates': []
    }

    cs0_csI_electron_rate['atom']['name'] = 'Cs'
    cs0_csI_electron_rate['atom']['symbol'] = 'Cs133'
    cs0_csI_electron_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
    cs0_csI_electron_rate['binding_units'] = 'eV'
    cs0_csI_electron_rate['reaction'] = 'Cs0 + e => CsI + 2e'
    cs0_csI_electron_rate['reaction_name'] = 'Primary ionization Cs via e'
    output['rates'].append(cs0_csI_electron_rate)

    cs0_csI_CX_rate['atom']['name'] = 'Cs'
    cs0_csI_CX_rate['atom']['symbol'] = 'Cs133'
    cs0_csI_CX_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
    cs0_csI_CX_rate['binding_units'] = 'eV'
    cs0_csI_CX_rate['reaction'] = 'Cs0 + H+ => CsI + H'
    cs0_csI_CX_rate['reaction_name'] = 'Primary ionization Cs via H CX'
    output['rates'].append(cs0_csI_CX_rate)

    csI_csN_electron_rate['atom']['name'] = 'Cs'
    csI_csN_electron_rate['atom']['symbol'] = 'Cs133'
    csI_csN_electron_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
    csI_csN_electron_rate['binding_units'] = 'eV'
    csI_csN_electron_rate['reaction'] = 'CsI + e => CsN + H'
    csI_csN_electron_rate['reaction_name'] = 'Secondary ionization Cs via e'
    output['rates'].append(csI_csN_electron_rate)

    cs0_csI_ion_rate['atom']['name'] = 'Cs'
    cs0_csI_ion_rate['atom']['symbol'] = 'Cs133'
    cs0_csI_ion_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
    cs0_csI_ion_rate['binding_units'] = 'eV'
    cs0_csI_ion_rate['reaction'] = 'Cs0 + H+ => Cs+ + H+ + e'
    cs0_csI_ion_rate['reaction_name'] = 'Ion impact ionization'
    output['rates'].append(cs0_csI_ion_rate)

    if impurities:
        cs0_csI_C6_rate['atom']['name'] = 'Cs'
        cs0_csI_C6_rate['atom']['symbol'] = 'Cs133'
        cs0_csI_C6_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
        cs0_csI_C6_rate['binding_units'] = 'eV'
        cs0_csI_C6_rate['reaction'] = 'Cs0 + C6+ => Cs+ + C6+ + e'
        cs0_csI_C6_rate['reaction_name'] = 'Carbon impact ionization'
        output['rates'].append(cs0_csI_C6_rate)

        cs0_csI_B5_rate['atom']['name'] = 'Cs'
        cs0_csI_B5_rate['atom']['symbol'] = 'Cs133'
        cs0_csI_B5_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
        cs0_csI_B5_rate['binding_units'] = 'eV'
        cs0_csI_B5_rate['reaction'] = 'Cs0 + B5+ => Cs+ + B5+ + e'
        cs0_csI_B5_rate['reaction_name'] = 'Boron impact ionization'
        output['rates'].append(cs0_csI_B5_rate)

        cs0_csI_C6_CX_rate['atom']['name'] = 'Cs'
        cs0_csI_C6_CX_rate['atom']['symbol'] = 'Cs133'
        cs0_csI_C6_CX_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
        cs0_csI_C6_CX_rate['binding_units'] = 'eV'
        cs0_csI_C6_CX_rate['reaction'] = 'Cs0 + C6+ => Cs+ + C5+'
        cs0_csI_C6_CX_rate['reaction_name'] = 'Carbon CX ionization'
        output['rates'].append(cs0_csI_C6_CX_rate)

        cs0_csI_B5_CX_rate['atom']['name'] = 'Cs'
        cs0_csI_B5_CX_rate['atom']['symbol'] = 'Cs133'
        cs0_csI_B5_CX_rate['binding_energy'] = np.array((Cs0_6s1, CsI_5p6))
        cs0_csI_B5_CX_rate['binding_units'] = 'eV'
        cs0_csI_B5_CX_rate['reaction'] = 'Cs0 + B5+ => Cs+ + B4+'
        cs0_csI_B5_CX_rate['reaction_name'] = 'Boron CX ionization'
        output['rates'].append(cs0_csI_B5_CX_rate)

    #--- Plotting.
    if plotflag:
        fig, ax = plt.subplots(1)
        xlabel = cs0_csI_electron_data['base_name'] + ' ['+ \
                 cs0_csI_electron_data['base_units'] + ']'
        ylabel = 'Cross section [' + \
                 cs0_csI_electron_data['units'] + ']'

        ax_options = { 'xlabel': xlabel,
                       'ylabel': ylabel,
                       'xscale': 'log',
                       'yscale': 'log',
                       'fontname': fontname,
                       'fontsize': fontsize
                     }

        ax.plot(cs0_csI_electron_data['base'],
                cs0_csI_electron_data['sigma'],
                label='$Cs^0+e \\rightarrow Cs^+$',
                **line_options)

        ax.plot(cs0_csI_CX_data['base'],
                cs0_csI_CX_data['sigma'],
                label='$Cs^0+H^+ \\rightarrow Cs^++H^0$',
                **line_options)

        ax.plot(csI_csII_electron_data['base'],
                csI_csII_electron_data['sigma'],
                label='$Cs^+ + e \\rightarrow Cs^{+2}$',
                **line_options)

        ax.plot(csI_csIII_electron_data['base'],
                csI_csIII_electron_data['sigma'],
                label='$Cs^+ + e \\rightarrow Cs^{+3}$',
                **line_options)

        ax.plot(csI_csIV_electron_data['base'],
                csI_csIV_electron_data['sigma'],
                label='$Cs^+ + e \\rightarrow Cs^{+4}$',
                **line_options)

        ax.plot(csI_csV_electron_data['base'],
                csI_csV_electron_data['sigma'],
                label='$Cs^+ + e \\rightarrow Cs^{+5}$',
                **line_options)

        ax.plot(cs0_csI_ion_data['base'],
                cs0_csI_ion_data['sigma'],
                label='$Cs^0 + H^+ \\rightarrow Cs^+ + H^+ +e$',
                **line_options)

        if impurities:
            ax.plot(cs0_csI_B5_data['base'],
                    cs0_csI_B5_data['sigma'],
                    label='$Cs^0 + B^{+5} \\rightarrow Cs^+ + B^{+5} + e$',
                    **line_options)

            ax.plot(cs0_csI_C6_data['base'],
                    cs0_csI_C6_data['sigma'],
                    label='$Cs^0 + C^{+6} \\rightarrow Cs^+ + C^{+6} + e$',
                    **line_options)

            ax.plot(cs0_csI_B5_H_data_CX['base'],
                    cs0_csI_B5_H_data_CX['sigma'],
                    label='$Cs^0 + B^{+5} \\rightarrow Cs^+ + B^{+4}$',
                    **line_options)

            ax.plot(cs0_csI_C6_H_data_CX['base'],
                    cs0_csI_C6_H_data_CX['sigma'],
                    label='$Cs^0 + C^{+6} \\rightarrow Cs^+ + C^{+5}$',
                    **line_options)

        ax = ssplt.axis_beauty(ax, ax_options)
        plt.legend()

        fig, ax = plt.subplots(1)
        xlabel = cs0_csI_electron_rate['base'][0]['name'] + ' ['+ \
                 cs0_csI_electron_rate['base'][0]['units']  + ']'
        ylabel = 'Reaction rate [' + \
                 cs0_csI_electron_rate['units'] + ']'

        ax_options = { 'xlabel': xlabel,
                       'ylabel': ylabel,
                       'xscale': 'log',
                       'yscale': 'log',
                       'fontname': fontname,
                       'fontsize': fontsize
                     }

        ax.plot(cs0_csI_electron_rate['base'][0]['data'],
                cs0_csI_electron_rate['rate'],
                label='$Cs^0+e \\rightarrow Cs^+$',
                **line_options)

        ax.plot(csI_csN_electron_rate['base'][0]['data'],
                csI_csN_electron_rate['rate'],
                label='$Cs^++e \\rightarrow Cs^{+N}$',
                **line_options)

        ax.plot(cs0_csI_CX_rate['base'][0]['data'],
                cs0_csI_CX_rate['rate'][:, 0],
                label='$Cs^0+H^+ \\rightarrow Cs^++H^0$',
                **line_options)

        ax.plot(cs0_csI_ion_rate['base'][0]['data'],
                cs0_csI_ion_rate['rate'][:, 0],
                label='$Cs^0+H^+ \\rightarrow Cs^++H^++ +e$',
                **line_options)

        if impurities:
            ax.plot(cs0_csI_C6_rate['base'][0]['data'],
                    cs0_csI_C6_rate['rate'][:, 0],
                    label='$Cs^0+C^{+6} \\rightarrow Cs^++C^{+6}+e$',
                    **line_options)

            ax.plot(cs0_csI_B5_rate['base'][0]['data'],
                    cs0_csI_B5_rate['rate'][:, 0],
                    label='$Cs^0+B^{+5} \\rightarrow Cs^++B^{+5}+e$',
                    **line_options)

            ax.plot(cs0_csI_C6_CX_rate['base'][0]['data'],
                    cs0_csI_C6_CX_rate['rate'][:, 0],
                    label='$Cs^0+C^{+6} \\rightarrow Cs^++C^{+5}$',
                    **line_options)

            ax.plot(cs0_csI_B5_CX_rate['base'][0]['data'],
                    cs0_csI_B5_CX_rate['rate'][:, 0],
                    label='$Cs^0+B^{+5} \\rightarrow Cs^++B^{+4}$',
                    **line_options)
        ax = ssplt.axis_beauty(ax, ax_options)
        plt.legend()

    return output


def createRbData(A, T: float, vel: float, plotflag: bool = True,
                 gridflag: bool = True, fontname = 'LM Roman 10',
                 fontsize = 11, line_options: dict = None,
                 impurities: bool = False):
    """
    Computes the reaction rates of the 85Rb or 87Rb.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param A: mass number to evaluate the table. Can only be 85 or 87
    @param T: temperatures to evaluate the reaction rates.
    @param vel: velocities to evaluate the reaction rates.
    @param plotflag: Flag to plot the reaction rates. True by default.
    @param gridflag: Flag to plot the grids in the plots. True by default.
    @param fontname: fontname to use in the plots.
    @param fontsize: fontsize for the plots.
    @para line_options: dictionary for the lines plotting.
    @returns output: returns the dictionary with the different reactions.
    """
    if A == 85:
        mass = alkMasses['Rb85']
        symbol = 'Rb85'

    elif A == 87:
        mass = alkMasses['Rb87']
        symbol = 'Rb87'
    else:
        raise Exception('Non valid mass number for stable Rubidium!')

    E = np.logspace(0, 7, num=101)
    # Ionization potentials of Rb
    chi = np.array((Rb0_5s1, RbI_4p6))
    occNum = np.array((1, 6))

    #--- Electron impact ionization.
    rb0_rbI_electron_data    = Lotz_cross_section(E, chi, occNum)
    rb0_rbI_CX_data          = Cs0_CX_H(E)
    rbI_rbN_electron_data   = RbI_RbN_electron(E)

    #--- Ion impact ionization.
    # We approximate the ion-impact ionization from the H formulae
    rb0_rbI_ion_data        = Wutte_scaling(H0_proton_imp_ion(E),
                                           Ubnew=Rb0_5s1)

    if impurities:
        #--- Impurity impact ionization.
        rb0_rbI_B5_Li_data      = Li_heavy_imp_ion(E, +5, Ub=Rb0_5s1)

        # For the impurity ionization we choose the worst-case scenario:
        rb0_rbI_B5_data = rb0_rbI_B5_Li_data

        # We approximate the C+6-impact ionization from the  Li formula
        rb0_rbI_C6_Li_data     = Li_heavy_imp_ion(E, +6, Ub=Rb0_5s1)

        # For the impurity ionization we choose the worst-case scenario:
        rb0_rbI_C6_data = rb0_rbI_C6_Li_data

        #--- Impurity CX ionization.
        # We approximate the B+5-CX ionization from the H formulae
        rb0_rbI_B5_H_data_CX    = Wutte_scaling(H0_B5_CX(E),
                                                Ubnew=Rb0_5s1)

        # We approximate the C+6-impact ionization from the H formulae
        rb0_rbI_C6_H_data_CX    = Wutte_scaling(H0_C6_CX(E),
                                                Ubnew=Rb0_5s1)

    #--- Computing the rates
    rb0_rbI_electron_rate = reactionRate(rb0_rbI_electron_data,
                                         massA = mass,
                                         massB = sspar.mass_electron_amu,
                                         T=T, vel=vel)

    rbI_rbN_electron_rate = reactionRate(rbI_rbN_electron_data,
                                         massA = mass,
                                         massB = sspar.mass_electron_amu,
                                         T=T, vel=vel)

    rb0_rbI_ion_rate = reactionRate(rb0_rbI_ion_data,
                                    massA = mass, massB = alkMasses['D'],
                                    T=T, vel=vel)

    rb0_rbI_CX_rate = reactionRate(rb0_rbI_CX_data,
                                    massA = mass, massB = alkMasses['D'],
                                    T=T, vel=vel)

    if impurities:
        rb0_rbI_C6_rate = reactionRate(rb0_rbI_C6_data,
                                       massA = mass, massB = alkMasses['C'],
                                       T=T, vel=vel)

        rb0_rbI_B5_rate = reactionRate(rb0_rbI_B5_data,
                                       massA = mass, massB = alkMasses['B'],
                                       T=T, vel=vel)

        rb0_rbI_C6_CX_rate = reactionRate(rb0_rbI_C6_H_data_CX,
                                          massA = mass, massB = alkMasses['C'],
                                          T=T, vel=vel)

        rb0_rbI_B5_CX_rate = reactionRate(rb0_rbI_B5_H_data_CX,
                                          massA = mass, massB = alkMasses['B'],
                                          T=T, vel=vel)

    #--- Creating the output variable:
    output = { 'date': str(date.today()),
    'species': 'Cs133',
    'species_symbol': '${}^{133}Cs$',
    'rates': []
    }

    rb0_rbI_electron_rate['atom']['name'] = 'Rb'
    rb0_rbI_electron_rate['atom']['symbol'] = symbol
    rb0_rbI_electron_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
    rb0_rbI_electron_rate['binding_units'] = 'eV'
    rb0_rbI_electron_rate['reaction'] = 'Rb0 + e => RbI + 2e'
    rb0_rbI_electron_rate['reaction_name'] = 'Primary ionization Rb'
    output['rates'].append(rb0_rbI_electron_rate)

    rbI_rbN_electron_rate['atom']['name'] = 'Rb'
    rbI_rbN_electron_rate['atom']['symbol'] = symbol
    rbI_rbN_electron_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
    rbI_rbN_electron_rate['binding_units'] = 'eV'
    rbI_rbN_electron_rate['reaction'] = 'CsI + e => CsN'
    rbI_rbN_electron_rate['reaction_name'] = 'Secondary ionization Rb'
    output['rates'].append(rbI_rbN_electron_rate)

    rb0_rbI_CX_rate['atom']['name'] = 'Rb'
    rb0_rbI_CX_rate['atom']['symbol'] = symbol
    rb0_rbI_CX_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
    rb0_rbI_CX_rate['binding_units'] = 'eV'
    rb0_rbI_CX_rate['reaction'] = 'CsI + e => CsN'
    rb0_rbI_CX_rate['reaction_name'] = 'Rb0 + H+ => RbI + H0'
    output['rates'].append(rb0_rbI_CX_rate)

    rb0_rbI_ion_rate['atom']['name'] = 'Rb'
    rb0_rbI_ion_rate['atom']['symbol'] = symbol
    rb0_rbI_ion_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
    rb0_rbI_ion_rate['binding_units'] = 'eV'
    rb0_rbI_ion_rate['reaction'] = 'Rb0 + H+ => Rb+ + H+ + e'
    rb0_rbI_ion_rate['reaction_name'] = 'Ion impact ionization'
    output['rates'].append(rb0_rbI_ion_rate)

    if impurities:
        rb0_rbI_C6_rate['atom']['name'] = 'Rb'
        rb0_rbI_C6_rate['atom']['symbol'] = symbol
        rb0_rbI_C6_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
        rb0_rbI_C6_rate['binding_units'] = 'eV'
        rb0_rbI_C6_rate['reaction'] = 'Rb0 + C6+ => Rb+ + C6+ + e'
        rb0_rbI_C6_rate['reaction_name'] = 'Carbon impact ionization'
        output['rates'].append(rb0_rbI_C6_rate)

        rb0_rbI_B5_rate['atom']['name'] = 'Rb'
        rb0_rbI_B5_rate['atom']['symbol'] = symbol
        rb0_rbI_B5_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
        rb0_rbI_B5_rate['binding_units'] = 'eV'
        rb0_rbI_B5_rate['reaction'] = 'Rb0 + B5+ => Rb+ + B5+ + e'
        rb0_rbI_B5_rate['reaction_name'] = 'Boron impact ionization'
        output['rates'].append(rb0_rbI_B5_rate)

        rb0_rbI_C6_CX_rate['atom']['name'] = 'Rb'
        rb0_rbI_C6_CX_rate['atom']['symbol'] = symbol
        rb0_rbI_C6_CX_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
        rb0_rbI_C6_CX_rate['binding_units'] = 'eV'
        rb0_rbI_C6_CX_rate['reaction'] = 'Rb0 + C6+ => Rb+ + C5+'
        rb0_rbI_C6_CX_rate['reaction_name'] = 'Carbon CX ionization'
        output['rates'].append(rb0_rbI_C6_CX_rate)

        rb0_rbI_B5_CX_rate['atom']['name'] = 'Rb'
        rb0_rbI_B5_CX_rate['atom']['symbol'] = symbol
        rb0_rbI_B5_CX_rate['binding_energy'] = np.array((Rb0_5s1, RbI_4p6))
        rb0_rbI_B5_CX_rate['binding_units'] = 'eV'
        rb0_rbI_B5_CX_rate['reaction'] = 'Rb0 + B5+ => Rb+ + B4+'
        rb0_rbI_B5_CX_rate['reaction_name'] = 'Boron CX ionization'
        output['rates'].append(rb0_rbI_B5_CX_rate)

    #--- Plotting.
    if plotflag:
        fig, ax = plt.subplots(1)
        xlabel = rb0_rbI_electron_data['base_name'] + ' ['+ \
        rb0_rbI_electron_data['base_units'] + ']'
        ylabel = 'Cross-section [' + \
        rb0_rbI_electron_data['units'] + ']'

        ax_options = { 'xlabel': xlabel,
           'ylabel': ylabel,
           'xscale': 'log',
           'yscale': 'log',
           'fontname': fontname,
           'fontsize': fontsize
         }

        ax.plot(rb0_rbI_electron_data['base'],
                rb0_rbI_electron_data['sigma'],
                label='$Rb^0+e \\rightarrow Rb^+$',
                **line_options)

        ax.plot(rb0_rbI_CX_data['base'],
                rb0_rbI_CX_data['sigma'],
                label='$Rb^0+H^+ \\rightarrow Rb^++H^0$',
                **line_options)

        ax.plot(rbI_rbN_electron_data['base'],
                rbI_rbN_electron_data['sigma'],
                label='$Rb^+ + e \\rightarrow Rb^{+2}$',
                **line_options)

        ax.plot(rb0_rbI_ion_data['base'],
                rb0_rbI_ion_data['sigma'],
                label='$Rb^0 + H^+ \\rightarrow Rb^+ + H^+ +e$',
                **line_options)

        if impurities:
            ax.plot(rb0_rbI_B5_data['base'],
                    rb0_rbI_B5_data['sigma'],
                    label='$Rb^0 + B^{+5} \\rightarrow Rb^+ + B^{+5} + e$',
                    **line_options)

            ax.plot(rb0_rbI_C6_data['base'],
                    rb0_rbI_C6_data['sigma'],
                    label='$Rb^0 + C^{+6} \\rightarrow Rb^+ + C^{+6} + e$',
                    **line_options)

            ax.plot(rb0_rbI_B5_H_data_CX['base'],
                    rb0_rbI_B5_H_data_CX['sigma'],
                    label='$Rb^0 + B^{+5} \\rightarrow Rb^+ + B^{+4}$',
                    **line_options)

            ax.plot(rb0_rbI_C6_H_data_CX['base'],
                    rb0_rbI_C6_H_data_CX['sigma'],
                    label='$Rb^0 + C^{+6} \\rightarrow Rb^+ + C^{+5}$',
                    **line_options)

        ax = ssplt.axis_beauty(ax, ax_options)
        plt.legend()

        fig, ax = plt.subplots(1)
        xlabel = rb0_rbI_electron_rate['base'][0]['name'] + ' ['+ \
                 rb0_rbI_electron_rate['base'][0]['units']  + ']'
        ylabel = 'Reaction rate [' + \
                 rb0_rbI_electron_rate['units'] + ']'

        ax_options = { 'xlabel': xlabel,
                       'ylabel': ylabel,
                       'xscale': 'log',
                       'yscale': 'log',
                       'fontname': fontname,
                       'fontsize': fontsize
                     }

        ax.plot(rb0_rbI_electron_rate['base'][0]['data'],
                rb0_rbI_electron_rate['rate'],
                label='$Rb^0+e \\rightarrow Rb^+$',
                **line_options)

        ax.plot(rbI_rbN_electron_rate['base'][0]['data'],
                rbI_rbN_electron_rate['rate'],
                label='$Rb^++e \\rightarrow Rb^{+N}$',
                **line_options)

        ax.plot(rb0_rbI_ion_rate['base'][0]['data'],
                rb0_rbI_ion_rate['rate'][:, 0],
                label='$Rb^0+H^+ \\rightarrow Rb^++H^+ + e$',
                **line_options)

        ax.plot(rb0_rbI_CX_rate['base'][0]['data'],
                rb0_rbI_CX_rate['rate'][:, 0],
                label='$Rb^0+H^+ \\rightarrow Rb^++H^0$',
                **line_options)

        if impurities:
            ax.plot(rb0_rbI_C6_rate['base'][0]['data'],
                    rb0_rbI_C6_rate['rate'][:, 0],
                    label='$Rb^0+C^{+6} \\rightarrow Rb^++C^{+6}+e$',
                    **line_options)

            ax.plot(rb0_rbI_B5_rate['base'][0]['data'],
                    rb0_rbI_B5_rate['rate'][:, 0],
                    label='$Rb^0+B^{+5} \\rightarrow Rb^++B^{+5}+e$',
                    **line_options)

            ax.plot(rb0_rbI_C6_CX_rate['base'][0]['data'],
                    rb0_rbI_C6_CX_rate['rate'][:, 0],
                    label='$Rb^0+C^{+6} \\rightarrow Rb^++C^{+5}$',
                    **line_options)

            ax.plot(rb0_rbI_B5_CX_rate['base'][0]['data'],
                    rb0_rbI_B5_CX_rate['rate'][:, 0],
                    label='$Rb^0+B^{+5} \\rightarrow Rb^++B^{+4}$',
                    **line_options)
        ax = ssplt.axis_beauty(ax, ax_options)
        plt.legend()

    return output
