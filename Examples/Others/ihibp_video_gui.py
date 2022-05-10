"""
Script to launch the iHIBP gui.
"""

import tkinter as tk
import Lib
from Lib.GUIs.iHIBP_exp_videos import app_ihibp_vid
from Lib.LibIO import ask_to_open

if __name__ == '__main__':
    iHIBP = Lib.dat.iHIBP
    root = tk.Tk()
    root.resizable(height=None, width=None)

    filetype = [('iHIBP strikeline map', '*.map')]

    a = app_ihibp_vid(root, shotnumber=40860,
                      path=None, signal_threshold=0)
    root.mainloop()
    root.destroy()