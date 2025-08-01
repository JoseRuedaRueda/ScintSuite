"""
Read the output files from the multipow program, written by Mike van Zeeland.
"""
import os
import matplotlib.pyplot as plt
from scipy.io import readsav


class MULTIPOW:
    """
    Class to read the output files from the multipow program, written by Mike van Zeeland.
    """

    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.read()

    def read(self):
        """
        Read the data from the file.
        """
        if os.path.exists(self.filename):
            self.data = readsav(self.filename)
        else:
            raise FileNotFoundError(f"File {self.filename} not found.")
    def ls(self):
        """
        List the variables in the file.
        """
        if self.data is not None:
            return list(self.data.keys())
        else:
            raise ValueError("No data loaded. Please read the file first.")
    
    def plotLine(self, axs=None, **kwargs):
        """
        Plot the data.
        """
        if axs is None:
            fig, axs = plt.subplots(4,1, sharex=True)
        else:
            fig = axs[0].figure
        induse = self.data['linepldat']['induse'][0] # To remove the repeated ECE channel
        # Plot the amplitude
        axs[0].plot(self.data['linepldat']['rhoece'][0][induse], 
                   self.data['ampfil'][induse],'o-', **kwargs)
        axs[0].set_ylabel('Amplitude')
        # plot the phase
        axs[1].plot(self.data['linepldat']['rhoece'][0][induse], 
                   self.data['phasfil'][induse], 'o-',**kwargs)
        axs[1].set_ylabel('Phase')
        # Plot the coherence
        axs[2].plot(self.data['linepldat']['rhoece'][0][induse], 
                   self.data['cohfil'][induse],'o-', **kwargs)
        axs[2].axhline(self.data['linepldat']['conf'], color='k', linestyle='--')
        axs[2].set_ylabel('Coherence')
        # plot the amp*phase
        axs[3].plot(self.data['linepldat']['rhoece'][0][induse], 
                   self.data['ampph'][induse], 'o-', **kwargs)
        axs[3].set_ylabel('Amp*cos(Phase)')
        axs[3].set_xlabel(r'$\rho_{tor}$')
        return axs
    
    def __getitem__(self, key):
        """
        Get the data for the given key.
        """
        if self.data is not None:
            if key in self.data:
                return self.data[key]
            else:
                raise KeyError(f"Key {key} not found in the data.")
        else:
            raise ValueError("No data loaded. Please read the file first.")
    
    