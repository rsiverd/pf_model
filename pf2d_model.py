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
def pf2dfit(x, y, z, nharm_x=0, nharm_y=0, nharm_xy_x=0, nharm_xy_y=0, npoly_x=1,
            npoly_y=1, npoly_xy_x=0, npoly_xy_y=0, weights=None, robust=False,
            M=sm.robust.norms.HuberT()):
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

    dm = np.zeros((npoly_x + npoly_y + npoly_xy_x * npoly_xy_y + 2 * (nharm_x +
                   nharm_y) + 4 * nharm_xy_x * nharm_xy_y + 1, z.size))
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
    npoly = npoly_x + npoly_y + npoly_xy_x * npoly_xy_y

    if npoly_x > 0:
        dm[:npoly_x] = (np.tile(x, (npoly_x, 1)).T ** np.array(range(1, npoly_x + 1))).T
    if npoly_y > 0:
        dm[npoly_x: npoly_x + npoly_y] = (np.tile(y, (npoly_y,1)).T ** np.array(range(1, npoly_y + 1))).T
    if npoly_xy_x > 0 and npoly_xy_y > 0:
        m, n = np.meshgrid(range(1, npoly_xy_x + 1), range(1, npoly_xy_y + 1))
        m = m.ravel()
        n = n.ravel()
        dm[npoly_x + npoly_y: npoly] = (np.tile(x, (npoly_xy_x * npoly_xy_y, 1)).T ** m).T
        dm[npoly_x + npoly_y: npoly] *= (np.tile(y, (npoly_xy_x * npoly_xy_y , 1)).T ** n).T

    # Fourier terms:
    if nharm_x > 0:
        modes_x = np.array(range(1, nharm_x + 1))
        dm[npoly: npoly + nharm_x] = np.sin(np.outer(modes_x, omegaTx))
        dm[npoly + nharm_x: npoly + nharm_x + nharm_x] = np.cos(np.outer(modes_x, omegaTx))

    if nharm_y > 0:
        modes_y = np.array(range(1, nharm_y + 1))
        dm[npoly + 2 * nharm_x: npoly + 2 * nharm_x + nharm_y] = np.sin(np.outer(modes_y, omegaTy))
        dm[npoly + 2 * nharm_x + nharm_y: npoly + 2 * nharm_x + 2 * nharm_y] = np.cos(np.outer(modes_y, omegaTy))

    # Do the cross terms
    if nharm_xy_x > 0 and nharm_xy_y > 0:
        modes_xy_x, modes_xy_y = np.meshgrid(range(1, nharm_xy_x + 1), range(1, nharm_xy_y + 1))
        modes_xy_x = modes_xy_x.ravel()
        modes_xy_y = modes_xy_y.ravel()

        dm[npoly + 2 * (nharm_x + nharm_y):-1: 4] = np.sin(np.outer(modes_xy_x, omegaTx)) * np.sin(np.outer(modes_xy_y, omegaTy))
        dm[npoly + 2 * (nharm_x + nharm_y) + 1:-1 : 4] = np.sin(np.outer(modes_xy_x, omegaTx)) * np.cos(np.outer(modes_xy_y, omegaTy))
        dm[npoly + 2 * (nharm_x + nharm_y) + 2:-1 : 4] = np.cos(np.outer(modes_xy_x, omegaTx)) * np.sin(np.outer(modes_xy_y, omegaTy))
        dm[npoly + 2 * (nharm_x + nharm_y) + 3:-1 : 4] = np.cos(np.outer(modes_xy_x, omegaTx)) * np.cos(np.outer(modes_xy_y, omegaTy))

    dm[-1] = np.ones(z.size)

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

    return coeffs

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


def pf2dcalc(coeffs, x, y, npoly_x=1, npoly_y=1, npoly_xy_x=0, npoly_xy_y=0, nharm_x=0,
             nharm_y=0, nharm_xy_x=0, nharm_xy_y=0):
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

    result = np.zeros(ny * nx)  # new Y values

    npoly = npoly_x + npoly_y + npoly_xy_x * npoly_xy_y
    # Add polynomial terms:
    if npoly_x > 0:
        result += (coeffs[:npoly_x] * (np.tile(x, (npoly_x,1))).T ** np.array(range(1, npoly_x + 1))).sum(axis=1)

    if npoly_y > 0:
        result += (coeffs[npoly_x:npoly_x + npoly_y] * (np.tile(y, (npoly_y,1))).T ** np.array(range(1, npoly_y + 1))).sum(axis=1)

    if npoly_xy_x > 0 and npoly_xy_y > 0:
        m, n = np.meshgrid(range(1, npoly_xy_x + 1), range(1, npoly_xy_y + 1))
        m = m.ravel()
        n = n.ravel()
        result += (coeffs[npoly_x + npoly_y: npoly] * (np.tile(x, (npoly_xy_x * npoly_xy_y,1))).T ** m * np.tile(y, (npoly_xy_x * npoly_xy_y, 1)).T ** n).sum(axis=1)

    # Add Fourier terms:
    if nharm_x > 0:
        modes_x = np.array(range(1, nharm_x + 1))
        result += (coeffs[npoly: npoly + nharm_x] * np.sin(np.outer(modes_x, omegaTx)).T).sum(axis=1)
        result += (coeffs[npoly + nharm_x: npoly + 2 * nharm_x] * np.cos(np.outer(modes_x, omegaTx)).T).sum(axis=1)

    if nharm_y > 0:
        modes_y = np.array(range(1, nharm_y + 1))
        result += (coeffs[npoly + 2 * nharm_x: npoly + 2 * nharm_x + nharm_y] * np.sin(np.outer(modes_y, omegaTy)).T).sum(axis=1)
        result += (coeffs[npoly + 2 * nharm_x + nharm_y: npoly + 2 * (nharm_x + nharm_y)] * np.cos(np.outer(modes_y, omegaTy)).T).sum(axis=1)

    # Do the cross terms
    if nharm_xy_x > 0 and nharm_xy_y > 0:
        modes_xy_x, modes_xy_y = np.meshgrid(range(1, nharm_xy_x + 1), range(1, nharm_xy_y + 1))
        modes_xy_x = modes_xy_x.ravel()
        modes_xy_y = modes_xy_y.ravel()

        result += (coeffs[npoly + 2 * (nharm_x + nharm_y) :-1: 4] * (np.sin(np.outer(modes_xy_x, omegaTx)) * np.sin(np.outer(modes_xy_y, omegaTy))).T).sum(axis=1)
        result += (coeffs[npoly + 2 * (nharm_x + nharm_y) + 1:-1 : 4] * (np.sin(np.outer(modes_xy_x, omegaTx)) * np.cos(np.outer(modes_xy_y, omegaTy))).T).sum(axis=1)
        result += (coeffs[npoly + 2 * (nharm_x + nharm_y) + 2:-1 : 4] * (np.cos(np.outer(modes_xy_x, omegaTx)) * np.sin(np.outer(modes_xy_y, omegaTy))).T).sum(axis=1)
        result += (coeffs[npoly + 2 * (nharm_x + nharm_y) + 3:-1 : 4] * (np.cos(np.outer(modes_xy_x, omegaTx)) * np.cos(np.outer(modes_xy_y, omegaTy))).T).sum(axis=1)

    result += coeffs[-1]
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
