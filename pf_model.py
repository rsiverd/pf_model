#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
#    Standalone polynomial + Fourier fitting module.
#
# Rob Siverd
# Created:       2014-09-30
# Last modified: 2015-06-24
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.1"

## Modules:
import os
import sys
import time
import numpy as np
#import scipy.optimize as opt
#import scipy.interpolate as stp
#from functools import partial
try:
    import statsmodels.api as sm
    __have_statsmodels__ = True
except ImportError:
    sys.stderr.write("statsmodels not found ... robust fitting disabled.\n")
    __have_statsmodels__ = False
#import theil_sen as ts

##--------------------------------------------------------------------------##
## Weighted least-squares fitting with numpy:
#def wlsq(X, y_vec, weights):
#   """Weighted least-squares fitting. X is design matrix."""
#   Xw = X * np.sqrt(weights)[:, np.newaxis]  # weighted design matrix
#   yw = y_vec * np.sqrt(weights)
#   return np.linalg.lstsq(Xw, yw)

##--------------------------------------------------------------------------##
## Short-hand for M-estimator selection:
#if __have_statsmodels__:
#   M_list = {}
#   M_list['huber'] = sm.robust.norms.HuberT()
#   M_list['tukey'] = sm.robust.norms.TukeyBiweight()

##--------------------------------------------------------------------------##
## Derivation:

# Fitting: A_n * sin(nωt + φ_n)



##--------------------------------------------------------------------------##
## Fit polynomial + Fourier series (statsmodels):
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
        dm_list.append(x**(i + 1.0))

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
    p_terms = coeffs[0:npoly+1]      # polynomial terms
    f_terms = coeffs[npoly+1::]      # Fourier terms
    amp_cos_phi = f_terms[0::2]      # coefficients of sin(nwt) terms
    amp_sin_phi = f_terms[1::2]      # coefficients of cos(nwt) terms

    # Amplitudes and phase shifts:
    f_ampli = np.sqrt(amp_cos_phi**2 + amp_sin_phi**2)
    f_shift = np.arctan2(amp_sin_phi, amp_cos_phi) % (2.0 * np.pi)
    return [p_terms, f_ampli, f_shift]

##--------------------------------------------------------------------------##
## Evaluate polynomial + Fourier model at specified positions:
def pfcalc(model, x):
   """Evaluate poly+Fourier model."""
   if (np.any(x < 0.0) or np.any(x > 1.0)):
      sys.stderr.write("x vector is out of bounds! (need 0 <= x < 1)\n")
      return None

   # Phase and results vectors:
   omegaT = 2.0 * np.pi * x            # convert to radians
   result = np.zeros_like(x)           # new Y values
   p_terms, f_ampli, f_shift = model   # parse model

   # Add polynomial terms:
   for pow, coeff in enumerate(p_terms):
      result += coeff * x**pow

   # Add Fourier terms:
   for h, (amp, shift) in enumerate(zip(f_ampli, f_shift)):
      phase = ((h + 1.0) * omegaT) + shift
      result += amp * np.sin(phase)

   return result

##--------------------------------------------------------------------------##
## Evaluate slope (derivative) of polynomial + Fourier model:
def pfslope(model, x):
   """Evaluate poly+Fourier model slope."""
   if (np.any(x < 0.0) or np.any(x > 1.0)):
      sys.stderr.write("x vector is out of bounds! (need 0 <= x < 1)\n")
      return None

   # Phase and results vectors:
   omegaT = 2.0 * np.pi * x            # convert to radians
   slopes = np.zeros_like(x)           # new slope values
   p_terms, f_ampli, f_shift = model   # parse model

   # Add polynomial terms:
   for pow, coeff in enumerate(p_terms):
      slopes += pow * coeff * x**(pow - 1.0)
      #if (pow > 0.0):
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
#  2015-06-24:
#     -- Increased __version__ to 0.2.1.
#     -- Fixed indentation in pffit().
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
