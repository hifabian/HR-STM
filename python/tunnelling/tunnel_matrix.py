# @author Hillebrand, Fabian
# @date   2018-2019

import numpy as np

from basis.wavefunction_abc import *
from tunnelling.chen_coeffs_abc import *

################################################################################
class TunnelMatrix():
  """!
    @brief Provides an object that computes the tunnel matrix entries.

    @attention This class will recompute the matrix entry when called again!
  """

  def __init__(self, chenCoeffs, wfn):
    """!
      @brief Constructor.

      @param chenCoeffs Coefficients for Chen's derivative rule.
      @param wfn        Evaluated wavefunction object.
    """
    self._dimGrid = wfn.dimGrid
    self._chenCoeffs = chenCoeffs
    self._wfn = wfn


  ##############################################################################
  ## Member variables
  @property
  def dimGrid(self):
    """! @return Dimension of one grid. """
    return self._dimGrid

  @property
  def noSpins(self):
    """! @brief Number of spins. """
    return self._noSpins

  @property
  def chenCoeffs(self):
    """! @brief Coefficient for Chen's derivative rule. """
    return self._chenCoeffs

  @property
  def wfn(self):
    """! @brief Wavefunction object. """
    return self._wfn


  ##############################################################################
  ## Member methods
  def __getitem__(self, idxTuple):
    """!
      @brief Computes and returns the tunnel matrix entry.

      @param idxTuple Multi-index in the form of a tuple:
                      (tunnelIdx, spinTipIdx, eTipIdx, spinSamIdx, eSamIdx).

      @return Tunnelling matrix entry.
    """
    # Indices
    tunnelIdx, spinTipIdx, eTipIdx, spinSamIdx, eSamIdx = idxTuple
    # Highest available derivatives
    noOrbs = min(self.chenCoeffs.noOrbs, self.wfn.noOrbsTip)
    noDer = (noOrbs+1)**2

    # Compute tunnel matrix element and return
    return np.einsum("o...,o...->...", \
      self.chenCoeffs[tunnelIdx,spinTipIdx,eTipIdx], \
      self.wfn[tunnelIdx,spinSamIdx,eSamIdx])

################################################################################
