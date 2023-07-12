"""
Script to launch the iHIBP gui.

This GUI will show the image of the iHIBP scintillator.
"""

import tkinter as tk
import Lib
from ScintSuite.GUIs.iHIBP_exp_videos import app_ihibp_vid
from ScintSuite.LibIO import ask_to_open

if __name__ == '__main__':
    iHIBP = ScintSuite.dat.iHIBP
    root = tk.Tk()
    root.resizable(height=None, width=None)

    filetype = [('iHIBP strikeline map', '*.map')]

    a = app_ihibp_vid(root, shotnumber=40860,
                      path=None, signal_threshold=0)
    root.mainloop()
    root.destroy()