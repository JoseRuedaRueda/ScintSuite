"""
Example to over-plot a theoretical frequency over a PySpecView spectogram
"""
import ScintSuite.as ss
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
pySpecViewFile = \
    '/afs/ipp/home/r/ruejo/PySpecViewOutput/AUG_41091_diag:_B31_sig:_B31-14.npz'

modeName = 'GAM'
shot = 41091
rho2plot = [0.2]
# -----------------------------------------------------------------------------
# --- Reading
# -----------------------------------------------------------------------------
mode = ss.mhd.MHDmode(shot)
spectra = ss.dat.pySpecView(pySpecViewFile)

# -----------------------------------------------------------------------------
# --- Plotting
# -----------------------------------------------------------------------------
ax = spectra.plot()
mode.plot(modeName, rho2plot, ax=ax, line_params={'alpha': 0.5})
