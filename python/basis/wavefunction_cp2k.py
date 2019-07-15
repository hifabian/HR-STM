# @author Hillebrand, Fabian
# @date   2018-2019

from . import wavefunction_abc

import time

import ctypes
import os
import numpy as np
import sysconfig as sc

from .interpolator import Interpolator

################################################################################
class WavefunctionHelp(wavefunction_abc.WavefunctionAbstract):
  """!
    @brief Wavefunction class that works on interpolation.

    Based on a wavefunction evaluated on a regular grid, the irregular grids
    are evaluated using interpolation.
    Derivatives are also provided where the differentiation acts on the
    interpolation.

    @attention Grids should be 3-tupel (or lists) with 3-dimensional arrays.
  """

  def __init__(self, **kwargs):
    """!
      @brief Initializer.

      @param wfn     Array containing the wavefunction evaluated on regular grid.
      @param regGrid Tupel or list defining the axis spacing.
      @param See WavefunctionAbstract
    """
    super().__init__(**kwargs, coefficients=[])
    self._wfnMatrix = kwargs['wfnMatrix']
    self._evalRegion = kwargs['evalRegion']
    self._dim = np.shape(self._wfnMatrix[0])[1:]
    self._regGrid = ( \
      np.linspace(self._evalRegion[0][0],self._evalRegion[0][1],self._dim[0]),
      np.linspace(self._evalRegion[1][0],self._evalRegion[1][1],self._dim[1]),
      np.linspace(self._evalRegion[2][0],self._evalRegion[2][1],self._dim[2]))
  
  ##############################################################################
  ## Member variables
  @property
  def wfnMatrix(self):
    """! @brief Returns the raw wavefunction data. """
    return self._wfnMatrix

  ##############################################################################
  ## OVERWRITTEN METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  def __getitem__(self, idxTupel):
    gridIdx, spinIdx, eigIdx = idxTupel
    # Check if already evaluated
    if self._wfn is not None \
      and self._gridC == gridIdx \
      and self._spinC == spinIdx \
      and self._eigC == eigIdx:
      return self._wfn
    # Storage container
    self._wfn = np.empty(((self.noOrbsTip+1)**2,)+self.dimGrid)

    # Create interpolator
    interp = Interpolator(self._regGrid, self._wfnMatrix[spinIdx][eigIdx])
    self._wfn[0] = interp(*self._grids[gridIdx])
    if self.noOrbsTip:
      self._wfn[1] = interp.gradient(*self._grids[gridIdx],1)
      self._wfn[2] = interp.gradient(*self._grids[gridIdx],2)
      self._wfn[3] = interp.gradient(*self._grids[gridIdx],3)
    self._gridC, self._spinC, self._eigC = idxTupel
    return self._wfn

################################################################################
