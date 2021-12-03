"""
Script to launch the iHIBP gui.
"""

import tkinter as tk
from Lib.GUIs.iHIBP_beam import appHIBP_beam
from Lib.LibParameters import iHIBP

if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(height=None, width=None)
    a = appHIBP_beam(root, origin=iHIBP['port_center'],
                     shotnumber=39810, time=0.50, diag='EQH')
    root.mainloop()
    root.destroy()