import os
import numpy as np
import tkinter as tk 
import Lib.BEP.LibBEP as rbep
from Lib.GUIs.BEP_gui import AppBEP_plot


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(height=None, width=None)
    a = AppBEP_plot(root, shotnumber = 38023, timeWindowOverlap=0.05)
    root.mainloop()
    root.destroy()
