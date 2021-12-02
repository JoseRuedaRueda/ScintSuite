"""
Script to launch the iHIBP gui.
"""

import tkinter as tk 
from Lib.GUIs.BEP_gui import iHIBP_beam
from Lib.LibParameters import iHIBP

if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(height=None, width=None)
    a = iHIBP_beam(root, origin=iHIBP['port_center'])
    root.mainloop()
    root.destroy()