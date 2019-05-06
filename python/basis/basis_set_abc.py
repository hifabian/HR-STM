# @author Hillebrand, Fabian
# @date   2018-2019

from abc import *

################################################################################
class BasisSetAbstract(ABC):
  """!
    @brief Provides an interface for a basis set object in the context of LCAO
           (Linear Combination of Atomic Orbitals). 

    The basis set consists of atomic orbitals centered on a specific atom. A 
    single object of this belongs to one atom-type only.

    Length unit is assumed to be given correctly. Generally speaking this is
    Bohr (convertion from Angstrom to Bohr should take place in the Wavefunction
    object).
  """

  ##############################################################################
  ## Class methods
  @abstractclassmethod
  def from_file(cls, fnames):
    """!
      @brief Creates a structure containing all basis functions for each atom
             in a system. Atoms of the same element share the atomic orbitals.

      @param fnames Input file or list of input files needed for contruction.
                    The concrete form depends on the implementation.

      @return A dictionary mapping an element symbol to a list of its basis
              functions.
    """
    return {"E" : [BasisSetAbstract()]}

  ##############################################################################
  ## Member variables
  @property
  @abstractmethod
  def noFunctions(self):
    """! @return Number of basis functions contained in this set. """
    return int

  @property
  @abstractmethod
  def lMin(self):
    """! @return Minimal value for angular momentum. """
    return int
    
  @property
  @abstractmethod
  def lMax(self):
    """! @return Maximal value for angular momentum. """
    return int

  ##############################################################################
  ## Member methods
  @abstractmethod
  def evaluate(self, grid, distSquared, basisCoeffs, orbsTip):
    """!
      @brief Evaluates the basis functions weighted by the given coefficients.

      @param grid        Grid values as seen from the atom in Bohr.
      @param distSquared Squared distance from atom to grid points in Bohr^2.
      @param basisCoeffs Weights for individual basis functions.
      @param orbsTip     Integer indicating maximal orbital on tip. 

      @return A list for each spin containing a 5-dimensional array.
              The form of the container is:
                [spinIdx][eneIdx, der, Z, Y, X]
    """
    return [np.zeros((0,0,0,0,0), dtype=float)]

################################################################################
