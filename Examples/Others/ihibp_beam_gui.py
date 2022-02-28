"""
Script to launch the iHIBP gui.
"""

import tkinter as tk
import Lib
from Lib.GUIs.iHIBP_beam import appHIBP_beam

if __name__ == '__main__':
    iHIBP = Lib.dat.iHIBP
    root = tk.Tk()
    root.resizable(height=None, width=None)
    a = appHIBP_beam(root, origin=iHIBP['port_center'],
                     shotnumber=39860, time=0.50, diag='EQH')
    root.mainloop()
    root.destroy()