"""Routines for the magnetic equilibrium"""
import numpy as np
import os

#from pyEquilibrium.equilibrium import equilibrium

def get_mag_field(shot: int, Rin: float, zin: float, time: float, **kwargs):
    """
    Get TCV magnetic field
    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
    Jesus Poley-Sanjuan - jesus.poley@epfl.ch
    Note: No extra arguments are expected, **kwargs is just included for
    compatibility of the call to this method in other databases (machines)
    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    Note: TCV-FILD Z position is always fixed at 0.05 [m] and the R position is the
    the port entrance position (where the tiles are) plus the insertion
    (which should be a negative value)
    :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T]
    """
    Z = zin#0.0500 # vertical FILD position [m]
    R = Rin#1.1376 + Rin # radial FILD position [m]
    # Create a temporary file named B.txt to store the local B [Br,Bphi,Bz] at
    # the FILD position (R, Z)
    a=f"matlab -nodisplay -r \"addpath('/home/poley/FILD/Jesus');" \
    f"eq=gdat({shot},'eqdsk','time',{time});" \
    f"equil=read_eqdsk(eq.eqdsk);" \
    f"R={R};Z={Z};position=1e3*[R*cos(0),R*sin(0),Z];" \
    f"B=extractDataEqdsk(equil,position,'Bcart');" \
    f"save('B.txt','B','-ASCII','-append');exit\""
    os.system(a) # Executing the a string
    # Read B.txt to get B field
    print('Reading magnetic field')
    f = open('B.txt') # Open txt file
    B = f.readlines()[0] # Read txt file containing B [T]
    br = float(B[2:16]) # take characters of Br and convert them to float
    bt = float(B[18:32]) # take characters of Br and convert them to float
    bz = float(B[35:48]) # ake characters of Br and convert them to float
    # Poloidal field is just obtained from Br and Bz
    bp = np.sqrt(br**2 + bz**2) # [T]
    # Delete B.txt file from current directory
    os.remove('B.txt')
    return br, bz, bt, bp















