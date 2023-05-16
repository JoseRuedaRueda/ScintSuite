"""
Custom models to perform fits with lmfit


Jose Rueda Rueda: jrrueda@us.es

Introduced in version
"""
import lmfit
import numpy as np
from lmfit.models import ExpressionModel

# -----------------------------------------------------------------------------
# %% Bivariant Gaussian
# -----------------------------------------------------------------------------
# Model taken from
# https://mathworld.wolfram.com/BivariateNormalDistribution.html
# just a scale constant added
fun = 'amp / (2.0*pi*sx*sy*sqrt(1-rho**2)) * ' + \
    'exp(-((x-mux)**2/sx**2 + (y-muy)**2/sy**2 - ' +\
    '2.0*rho*(x-mux)*(y-muy)/sx/sy)/(2.0*(1-rho**2)))'
BivariateNormalDistribution = ExpressionModel(fun, independent_vars=['x', 'y'])

def guessParamsBivariateNormalDistribution(x, y, z):
    """
    Guess the parameters for the bivariate normal distribution

    formula of weiugthed standard deviation taken from nist:
    https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    assume that N' is much larger than 1, so the denominator simplifies
    """
    # ---- Calculate the weighted means
    mux = (x*z).sum() / z.sum()
    muy = (y*z).sum() / z.sum()
    # ---- Calculate the weighted std
    sx = np.sqrt((z * (x - mux)**2).sum() / z.sum())
    sy = np.sqrt((z * (y - muy)**2).sum() / z.sum())
    # ---- Calculate the amplitude
    amp = z.sum()
    if amp > 0:
        maxamp = 10*amp
        minamp = 0.0
    else:
        minamp = 10*amp
        maxamp = 0.0
    # ---- correlation
    # For the correlation, just leave it to zero, and the model will fit
    rho = 0
    # ---- Create params
    # init some dummy values
    p = BivariateNormalDistribution.make_params(amp=7.0e8, sx=0.05, sy=5,
                                                mux=1.7, muy=60.0, rho=0.5)
    # fill the proper values
    p['rho'].set(0.0, min=-1.0, max=1.0)
    p['amp'].set(amp, min=minamp, max=maxamp)
    p['sx'].set(sx, min=0.1*sx, max=10*sx)
    p['sy'].set(sy, min=0.1*sy, max=10*sy)
    p['muy'].set(muy, min=0.5*muy, max=1.5*muy)
    p['mux'].set(mux, min=0.5*mux, max=1.5*mux)
    return p
