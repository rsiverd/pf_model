#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
#    Standalone polynomial + Fourier fitting module.
#
# Rob Siverd
# Created:       2014-09-30
# Last modified: 2015-03-09
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

# Current version:
__version__ = "0.3.0"

# Modules:
import sys
import numpy as np
import warnings
# import scipy.optimize as opt
# import scipy.interpolate as stp
# from functools import partial
try:
    import statsmodels.api as sm
    __have_statsmodels__ = True
except ImportError:
    warnings.warn("statsmodels not found ... robust fitting disabled.\n",
                  ImportWarning)
    __have_statsmodels__ = False
# import theil_sen as ts

# #--------------------------------------------------------------------------##
# Weighted least-squares fitting with numpy:
# def wlsq(X, y_vec, weights):
#   """Weighted least-squares fitting. X is design matrix."""
#   Xw = X * np.sqrt(weights)[:, np.newaxis]  # weighted design matrix
#   yw = y_vec * np.sqrt(weights)
#   return np.linalg.lstsq(Xw, yw)

# #--------------------------------------------------------------------------##
# Short-hand for M-estimator selection:
# if __have_statsmodels__:
#   M_list = {}
#   M_list['huber'] = sm.robust.norms.HuberT()
#   M_list['tukey'] = sm.robust.norms.TukeyBiweight()

# #--------------------------------------------------------------------------##
# Derivation:

# Fitting: A_n * sin(nωt + φ_n)


# #--------------------------------------------------------------------------##
# Fit polynomial + Fourier series (statsmodels):
def pffit(x, y, nharm, npoly=0, weights=None,
          robust=False, M=sm.robust.norms.HuberT()):
    """Polynomial + Fourier fitting routine."""
    if (np.any(x < 0.0) or np.any(x > 1.0)):
        sys.stderr.write("x vector is out of bounds! (need 0 < x < 1)\n")
        return None
    omegaT = 2.0 * np.pi * x
    dm_list = [np.ones_like(x)]

    # Disable robust if statsmodels missing:
    if robust and (not __have_statsmodels__):
        sys.stderr.write("Robust fitting disabled (statsmodels not found)!\n")
        robust = False

    # Note bogus weight vector:
    if isinstance(weights, np.ndarray) and (y.size != weights.size):
        sys.stderr.write("Ignoring weights vector (wrong size)!\n")
        weights = None

    # Polynomial terms:
    for i in range(npoly):
        dm_list.append(x ** (i + 1.0))

    # Fourier terms:
    for mode in range(1, nharm + 1):
        dm_list.append(np.sin(mode * omegaT))
        dm_list.append(np.cos(mode * omegaT))

    # Make design matrix (with weights if needed):
    if isinstance(weights, np.ndarray) and (y.size == weights.size):
        design_matrix = np.array(dm_list).T * np.sqrt(weights)[:, np.newaxis]
        y_vals = y * np.sqrt(weights)
    else:
        design_matrix = np.array(dm_list).T
        y_vals = y

    # Fit with numpy or statsmodels:
    if robust:
        result = sm.RLM(y_vals, design_matrix, M=M).fit()
        coeffs = result.params
    else:
        answer = np.linalg.lstsq(design_matrix, y_vals)
        coeffs = answer[0]

    # Parse parameters:
    p_terms = coeffs[0:npoly + 1]  # polynomial terms
    f_terms = coeffs[npoly + 1::]  # Fourier terms
    amp_cos_phi = f_terms[0::2]  # coefficients of sin(nwt) terms
    amp_sin_phi = f_terms[1::2]  # coefficients of cos(nwt) terms

    # Amplitudes and phase shifts:
    f_ampli = np.sqrt(amp_cos_phi ** 2 + amp_sin_phi ** 2)
    f_shift = np.arctan2(amp_sin_phi, amp_cos_phi) % (2.0 * np.pi)
    return [p_terms, f_ampli, f_shift]


# Fit polynomial + Fourier series (statsmodels):
def pf2dfit(x, y, z, nharm_x, nharm_y, nharm_xy, npoly_x=1, npoly_y=1, npoly_xy=0,
            weights=None, robust=False, M=sm.robust.norms.HuberT()):
    """Polynomial + Fourier fitting routine."""
    if (np.any(x < 0.0) or np.any(x > 1.0)):
        # Rescale the x and y inputs
        x = (x - x.min()) / (x.max() - x.min())
    
    if (np.any(y < 0.0) or np.any(y > 1.0)):
        # Rescale the x and y inputs
        y = (y - y.min()) / (y.max() - y.min())
                
    x2d, y2d = np.meshgrid(x, y)
    # unravel the x,y, and z arrays
    x = x2d.ravel()
    y = y2d.ravel()
    z = z.ravel()
    
    omegaTx = 2.0 * np.pi * x
    omegaTy = 2.0 * np.pi * y
    omegaTxy = 2.0 * np.pi * x * y
    
    dm = np.zeros((npoly_x + npoly_y + npoly_xy + nharm_x + nharm_y + nharm_xy,
                   z.size))
    # Disable robust if statsmodels missing:
    if robust and (not __have_statsmodels__):
        warnings.warn("Robust fitting disabled (statsmodels not found)!\n",
                      ImportWarning)
        robust = False

    # Note bogus weight vector:
    if isinstance(weights, np.ndarray) and (z.size != weights.size):
        warnings.warn("Ignoring weights vector (wrong size)!\n")
        weights = None

    # Polynomial terms:
    npoly = npoly_x + npoly_y + npoly_xy
    if npoly_x > 0:
        dm[:npoly_x + 1] = np.tile(x, npoly_x) ** np.array(range(1, npoly_x + 1))
    if npoly_y > 0:
        dm[npoly_x + 1: npoly_x + npoly_y + 1] = np.tile(y, npoly_y) ** np.array(range(1, npoly_y + 1))
    if npoly_xy > 0:
        dm[npoly_x + npoly_y + 1: npoly + 1] = np.tile(x * y, npoly_xy) ** np.array(range(1, npoly_xy + 1))
    
    # Fourier terms:
    modes_x = np.array(range(1, nharm_x + 1))
    dm[npoly + 1: npoly + nharm_x + 1] = np.sin(np.outer(modes_x, omegaTx))
    dm[npoly + nharm_x + 1: npoly + nharm_x + nharm_x + 1] = np.cos(np.outer(modes_x, omegaTx))
    
    modes_y = np.array(range(1, nharm_y + 1))
    dm[npoly + 2 * nharm_x + 1: npoly + 2 * nharm_x + nharm_y + 1] = np.sin(np.outer(modes_y, omegaTy))
    dm[npoly + 2 * nharm_x + nharm_y + 1: npoly + 2 * nharm_x + 2 * nharm_y + 1] = np.cos(np.outer(modes_y, omegaTy))
    
    modes_xy = np.array(range(1, nharm_xy + 1))
    dm[npoly + 2 * (nharm_x + nharm_y) + 1: npoly + 2 * (nharm_x + nharm_y) + nharm_xy + 1] = np.sin(np.outer(modes_xy, omegaTxy))
    dm[npoly + 2 * (nharm_x + nharm_y) + nharm_xy + 1:] = np.cos(np.outer(modes_xy, omegaTxy))
    
    # Make design matrix (with weights if needed):
    if weights is not None:
        design_matrix = dm.T * np.sqrt(weights)[:, np.newaxis]
        z_vals = z * np.sqrt(weights)
    else:
        design_matrix = dm.T
        z_vals = z

    # Fit with numpy or statsmodels:
    if robust:
        result = sm.RLM(z_vals, design_matrix, M=M).fit()
        coeffs = result.params
    else:
        answer = np.linalg.lstsq(design_matrix, z_vals)
        coeffs = answer[0]

    # Parse parameters:
    px_terms = coeffs[0:npoly_x]  # polynomial terms
    py_terms = coeffs[npoly_x:npoly_x + npoly_y]  # polynomial terms
    pxy_terms = coeffs[npoly_x + npoly_y : npoly]
    
    fx_sin = coeffs[npoly: npoly + nharm_x]
    fx_cos = coeffs[npoly + nharm_x: npoly + 2 * nharm_x]
    fy_sin = coeffs[npoly + 2 * nharm_x: npoly + 2 * nharm_x + nharm_y]
    fy_cos = coeffs[npoly + 2 * nharm_x + nharm_y : npoly + 2 * nharm_x + 2 * nharm_y]
    fxy_sin = coeffs[npoly + 2 * nharm_x + 2 * nharm_y: npoly + 2 * nharm_x + 2 * nharm_y + nharm_xy]
    fxy_cos = coeffs[npoly + 2 * nharm_x + 2 * nharm_y + nharm_xy: ]
    
    return [px_terms, py_terms, pxy_terms, fx_sin, fx_cos, fy_sin, fy_cos, fxy_sin, fxy_cos]

# #--------------------------------------------------------------------------##
# Evaluate polynomial + Fourier model at specified positions:

def pfcalc(model, x):
    """Evaluate poly+Fourier model."""
    if (np.any(x < 0.0) or np.any(x > 1.0)):
        sys.stderr.write("x vector is out of bounds! (need 0 <= x < 1)\n")
        return None

    # Phase and results vectors:
    omegaT = 2.0 * np.pi * x  # convert to radians
    result = np.zeros_like(x)  # new Y values
    p_terms, f_ampli, f_shift = model  # parse model

    # Add polynomial terms:
    for pow, coeff in enumerate(p_terms):
        result += coeff * x ** pow

    # Add Fourier terms:
    for h, (amp, shift) in enumerate(zip(f_ampli, f_shift)):
        phase = ((h + 1.0) * omegaT) + shift
        result += amp * np.sin(phase)

    return result


def pf2dcalc(model, x, y):
    """Evaluate poly+Fourier model."""
    if (np.any(x < 0.0) or np.any(x > 1.0)):
        # Rescale the x and y inputs
        x = (x - x.min()) / (x.max() - x.min())
    
    if (np.any(y < 0.0) or np.any(y > 1.0)):
        # Rescale the x and y inputs
        y = (y - y.min()) / (y.max() - y.min())
            
    nx = len(x)
    ny = len(y)
    
    x2d, y2d = np.meshgrid(x, y)
    # unravel the x,y, and z arrays
    x = x2d.ravel()
    y = y2d.ravel()

    # Phase and results vectors:
    # convert to radians    
    omegaTx = 2.0 * np.pi * x
    omegaTy = 2.0 * np.pi * y
    omegaTxy = 2.0 * np.pi * x * y

    result = np.zeros(ny * nx)  # new Y values
    px_terms, py_terms, pxy_terms, fx_sin, fx_cos, fy_sin, fy_cos, fxy_sin, fxy_cos = model  # parse model

    # Add polynomial terms:
    if len(px_terms) > 0:
        result += px_terms * np.tile(x, len(px_terms)) ** np.array(range(1, len(px_terms) + 1))
    if len(py_terms) > 0:
        result += py_terms * np.tile(y, len(py_terms)) ** np.array(range(1, len(py_terms) + 1))
    if len(pxy_terms) > 0:
        result += pxy_terms * np.tile(x * y, len(pxy_terms)) ** np.array(range(1, len(pxy_terms) + 1))

    # Add Fourier terms:
    if len(fx_sin) > 0:
        modes_x = np.array(range(1, len(fx_sin) + 1))
        result += fx_sin * np.sin(np.outer(modes_x, omegaTx))
        result += fx_cos * np.cos(np.outer(modes_x, omegaTx))    
    
    if len(fy_sin) > 0:
        modes_y = np.array(range(1, len(fxy_sin) + 1))
        result += fy_sin * np.sin(np.outer(modes_y, omegaTy))
        result += fy_cos * np.cos(np.outer(modes_y, omegaTy))
    
    if len(fxy_sin) > 0:
        modes_xy = np.array(range(1, len(fxy_sin) + 1))
        result += fxy_sin * np.sin(np.outer(modes_xy, omegaTxy))
        result += fxy_cos * np.cos(np.outer(modes_xy, omegaTxy))

    return result.reshape((ny, nx))

# #--------------------------------------------------------------------------##
# Evaluate slope (derivative) of polynomial + Fourier model:


def pfslope(model, x):
    """Evaluate poly+Fourier model slope."""
    if (np.any(x < 0.0) or np.any(x > 1.0)):
        sys.stderr.write("x vector is out of bounds! (need 0 <= x < 1)\n")
        return None

    # Phase and results vectors:
    omegaT = 2.0 * np.pi * x  # convert to radians
    slopes = np.zeros_like(x)  # new slope values
    p_terms, f_ampli, f_shift = model  # parse model

    # Add polynomial terms:
    for pow, coeff in enumerate(p_terms):
        slopes += pow * coeff * x ** (pow - 1.0)
        # if (pow > 0.0):
        #   slopes += pow * coeff * x**(pow - 1.0)

    # Add Fourier terms:
    for h, (amp, shift) in enumerate(zip(f_ampli, f_shift)):
        phase = ((h + 1.0) * omegaT) + shift
        slopes += (h + 1.0) * amp * np.cos(phase)

    return slopes


######################################################################
# CHANGELOG (pf_model.py):
#---------------------------------------------------------------------
#
#  2014-10-06:
#     -- Increased __version__ to 0.2.0.
#     -- Added pfslope() derivative evaluation routine.
#
#  2014-10-01:
#     -- Increased __version__ to 0.1.0.
#     -- Now use np.linalg.lstsq() for all non-robust fits (~3x faster).
#     -- Simplified parsing of poly/harm coefficients.
#
#  2014-09-30:
#     -- Increased __version__ to 0.0.1.
#     -- First created pf_model.py.
#
