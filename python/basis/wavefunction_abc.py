# @author Hillebrand, Fabian
# @date   2018-2019

from abc import *

import numpy as np

################################################################################
class WavefunctionAbstract(ABC):
  """!
    @brief Provides an interface for a wavefunction object.

    This class provides access to the wavefunctions via a bracket operator.

    The structure provided takes the form:
      [gridIdx,spinIdx,eneIdx][derIdx,pointIdx]
    or
      [gridIdx,spinIdx,eneIdx][derIdx,zIdx,yIdx,xIdx]
    where
      gridIdx  Index of grid.
      spinIdx  Index of spin for wavefunction.
      eneIdx   Index of eigenenergy corresponding to wavefunction.
      derIdx   Integer indicating which differential operator is applied
               to wavefunction.
      pointIdx Flattened coordinate for grid point.
    The differential operators refer to the tip-orbital and are in order {I,
    1/kappa*d/dy, 1/kappa*d/dz, 1/kappa*d/dx} corresponding to {s, py, pz, px}.

    @note The input parameters are assumed to be in Angstrom and electron Volt.
  """

  def __init__(self, **kwargs):
    """!
      @brief Default initializer. Parameters should be pass-by-name.

      @param noOrbsTip    Maximal orbital on tip used for determining the
                          differential operators evaluated.
      @param eigs         Eigenenergies.
      @param coefficients Basis function coefficients.
      @param atoms        Atoms-object.
    """
    self._noOrbsTip = kwargs['noOrbsTip']
    self._eigs = kwargs['eigs']
    self._coefficients = kwargs['coefficients']
    self._atoms = kwargs['atoms']
    self._grids = None
    self._wfn = None

  ##############################################################################
  ## Member variables
  @property
  def noGrids(self):
    """! @return Total number of grids. """
    if self._grids is not None:
      return len(self._grids)
    else:
      raise AttributeError("No grid is set yet!")
  @property
  def grids(self):
    """!
      @brief Provides access to grids.

      @return Returns evaluation grids.
    """
    return self._grids
  @property
  def dimGrid(self):
    """! @return Dimension of one grid. Throws an AttributeError if not set. """
    if self._grids is not None and self._grids != []:
      return np.shape(self._grids[0])[:-1]
    else:
      raise AttributeError("No grid is set yet!")

  @property
  def noSpins(self):
    """! @return Number of spins. """
    return len(self._coefficients)

  @property
  def noOrbsTip(self):
    """! @return Maximal tip orbital. 0 for s-orbital, 1 for s- and p-orbitals. """
    return self._noOrbsTip

  @property
  def noAtoms(self):
    """! @return Number of atoms. """
    return len(self._atoms)
  @property
  def atoms(self):
    """! @return Atom object. """
    return self._atoms

  @property
  def noEigs(self):
    """! @return A list containing the number of energies per spin. """
    return [len(self._eigs[spinIdx]) for spinIdx in range(self.noSpins)]
  @property
  def eigs(self):
    """! @return A list containing all eigenenergies for each spin. """
    return self._eigs
  @property
  def coefficients(self):
    """! @return The coefficients. """
    return self._coefficients

  ##############################################################################
  ### Member methods
  def __getitem__(self, idxTupel):
    """!
      @brief Provides bracket-operator access.

      @param idxTupel Tupel of indices (gridIdx, spinIdx, eigIdx).

      @return Wavefunctions on a specific grid at specific eigenenergy.
    """
    gridIdx, spinIdx, eigIdx = idxTupel
    return self._wfn[gridIdx][spinIdx][eigIdx]

  def setGrids(self, grids):
    """!
      @brief Sets grids for wavefunction.

      @param grids Grids that are set.
    """
    self._grids = grids

################################################################################
