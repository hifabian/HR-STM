# @author Hillebrand, Fabian
# @date   2018-2019

from abc import *

import numpy as np

################################################################################
class ChenCoeffsAbstract(ABC):
  """!
    @brief Provides an interface for a Chen's derivative rule coefficients 
           object.

    This class provides access to the coefficients via bracket operators.
    The following structure is assumed: 
      [tunnelIdx,spinIdx,eneIdx][derIdx,zIdx,yIdx,xIdx]
    where
    gridIdx         Index of grid.
    spinIdx         Index of spin for wavefunction.
    eneIdx          Index of eigenenergy corresponding to wavefunction.
    derIdx          Tip orbital {s, py, pz, px}.
    zIdx,yIdx,xIdx  Coordinate for grid point.

    @note The input parameters are assumed to be in Angstrom and electron Volt.
  """

  def __init__(self, **kwargs):
    """
      @brief Initializer. Parameters should be pass-by-name.

      @param noOrbs Maximal orbital on tip.
      @param single A list of coefficients in form of a list of matrices 
                    for each spin.
      @param eigs   A list of arrays containing eigenvalues for each spin.
      @param rotate Flag indicating if orbitals ought to be rotated.

      @note The grids should be of the form [z,y,x,3].
    """
    self._noOrbs = kwargs['noOrbs']
    self._singles = kwargs['singles']
    self._eigs = kwargs['eigs']
    self._rotate = kwargs['rotate']
    self._grids = None
    self._coeffs = None

  ##############################################################################
  ## Member variables
  @property
  def dimGrid(self):
    """! @return Dimension of one grid. Throws an AttributeError if not set. """
    if self._grids != []:
      return np.shape(self._grids[0])[1:]
    else:
      raise AttributeError("No grid is set yet!")

  @property
  def noSpins(self):
    """! @brief Number of spins. """
    return len(self._eigs)

  @property
  def noTunnels(self):
    """! @brief Number of tunnelings. """
    return len(self._grids)-1

  @property
  def noEigs(self):
    """! @brief A list containing the number of energies per spin. """
    return [len(self._eigs[spinIdx]) for spinIdx in range(self.noSpins)]
  @property
  def eigs(self):
    """! @return A list containing all eigenvalues for each spin. """
    return self._eigs

  @property
  def noOrbs(self):
    """! @brief Maximal orbitals. """
    return self._noOrbs

  @property
  def singles(self):
    """! @brief Returns the untransformed Chen's coefficients. """
    return self._singles
  @singles.setter
  def singles(self, singles):
    """! @brief Set untransformed Chen's coefficients. """
    self._singles = singles

  @property
  def rotate(self):
    """! @brief Returns the rotation flag. """
    return self._rotate

  ##############################################################################
  ## Member methods
  @abstractmethod
  def __getitem__(self, idxTuple):
    """!
      @brief Provides bracket-operator access.

      In particular, this method invokes a computation on the grid. This is
      done here to save memory at the cost of potential recomputation.

      @param idxTuple Tupel of indices [tunnelIdx, spinIdx, eigIdx]

      @return Coefficient for Chen's derivative rule for each grid point.
    """
    tunnelIdx, spinIdx, eigIdx = idxTupel
    return self._coeffs[tunnelIdx][spinIdx][eigIdx]

  def setGrids(self, grids):
    """!
      @brief Sets grids for wavefunction.

      @param grids Grids that are set.
    """
    self._grids = grids

################################################################################
