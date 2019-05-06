# @author Hillebrand, Fabian
# @date   2018-2019

from . import wavefunction_abc

import numpy as np
import copy as cp

################################################################################
class WavefunctionTest(wavefunction_abc.WavefunctionAbstract):
  """!
    @brief Simple wavefunction useful for debugging.
           The computed wavefunction has no physical meaning and is simply
           meant as a debugging tool. Concrete implementation is to be changed.
  """

  ##############################################################################
  ## OVERWRITTEN METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  ### Member methods
  def __getitem__(self, idxTupel):
    """!
      @brief Provides bracket-operator access.

      @param idxTupel Tupel of indices (gridIdx, spinIdx, eigIdx).

      @return Wavefunctions on a specific grid at specific eigenenergy.
    """
    gridIdx, spinIdx, eigIdx = idxTupel

    wfn = np.zeros(((self.noOrbsTip+1)**2,)+self.dimGrid)

    # WFN(x,y,z) = 50-z
    wfn[0] = 50.0-self._grids[gridIdx][:,2]
    if self.noOrbsTip > 0:
      # dy -> 0
      wfn[1] = 0.0
      # dz -> -1
      wfn[2] = -1.0
      # dx -> 0
      wfn[3] = 0.0

    return wfn

################################################################################
