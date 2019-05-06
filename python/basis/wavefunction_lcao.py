# @author Hillebrand, Fabian
# @date   2018-2019

from . import wavefunction_abc

import time
import copy as cp
import numpy as np
from operator import add

# ------------------------------------------------------------------------------

# Constants
angstromToBohr = 1.0/0.52917721067

################################################################################
class WavefunctionLCAO(wavefunction_abc.WavefunctionAbstract):
  """!
    @brief Realization of abstract wavefunction class for the LCAO approach.
  """

  def __init__(self, **kwargs):
    """!
      @brief Initializer.

      @param basisSets    Dictionary for basis sets on all elements.
      @param See WavefunctionAbstract.
    """
    super().__init__(**kwargs)
    self._wfn = []
    self._basisSets = cp.deepcopy(kwargs['basisSets'])
    self._computed = False

  ##############################################################################
  ## Member methods
  def _atomic_orbital(self, localGrid, aoBasisSets, aoCoeffs, noOrbsTip):
    """!
      @brief Evaluates contribution from basis functions on a specific atom.

      @param localGrid   Grid with atom as origin in Bohr.
      @param aoBasisSets A container for the basis sets on atom.
      @param aoCoeffs    Coefficients for basis functions on atom.
      @param noOrbsTip   Maximal orbital on tip used for determining the
                         differential operators evaluated.

      @return A container containing the atomic orbitals on the grid. A specific
              atomic orbital can be accessed using [spinIdx][eneIdx,der,Z,Y,X].
    """
    noSpins = self.noSpins
    noEigs = self.noEigs
    dimGrid = np.shape(localGrid)[:-1]
    noDer = (noOrbsTip+1)**2
    # Distance from origin squared
    distSquared = localGrid[:,0]**2+localGrid[:,1]**2+localGrid[:,2]**2
    # Storage container
    orbitals = [np.zeros((noEigs[spinIdx],noDer,)+dimGrid) for spinIdx in range(noSpins)]

    coeffIdx = 0
    for basisIdx, basisSet in enumerate(aoBasisSets):
      noBasis = basisSet.noFunctions
      # Extract relevant coefficients 
      coeffs = [aoCoeffs[spinIdx][:,coeffIdx:coeffIdx+noBasis] for spinIdx in range(noSpins)]
      # Add contribution
      orbitals = list(map(add, orbitals, \
        basisSet.evaluate(localGrid, distSquared, coeffs, noOrbsTip)))
      coeffIdx += noBasis
    return orbitals

  def _compute(self):
    """! @brief Evaluates the wavefunctions. """
    start0 = time.time()
    noSpins = self.noSpins
    noEigs = self.noEigs
    noAtoms = self.noAtoms
    dimGrid = self.dimGrid
    noDer = (self.noOrbsTip+1)**2

    for gridIdx in range(self.noGrids):
      self._wfn.append([np.zeros((noEigs[spinIdx],noDer,)+dimGrid) for spinIdx in 
        range(noSpins)])
      # Dealing with LCAO: Compute wavefunction atom-wise
      for atomIdx in range(noAtoms):
        # NOTE Print progress for debugging
        print("atomIdx="+str(atomIdx+1)+"/"+str(noAtoms)+"\r", end='', flush=True)
        elem = self._atoms[atomIdx].symbol
        pos = self._atoms[atomIdx].position
        # Grid with origin on this atom
        localGrid = (self._grids[gridIdx]-pos)*angstromToBohr
        # Coefficients for basis on specific atom
        coeffs = [self._coefficients[spinIdx][atomIdx] for spinIdx in \
          range(noSpins)]
        # Adding atomic orbital contribution
        self._wfn[gridIdx] = list(map(add, self._wfn[gridIdx], \
          self._atomic_orbital(localGrid, self._basisSets[elem], coeffs, \
          self.noOrbsTip)))
      # NOTE Print progress for debugging
      print("")
    end0 = time.time()
    print("Wavefunction took {} seconds".format(end0-start0))

    self._computed = True

  ##############################################################################
  ## OVERWRITTEN METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  ## Member methods
  def setGrids(self, grids):
    self._grids = grids
    if not self._computed:
      self._compute()

################################################################################
