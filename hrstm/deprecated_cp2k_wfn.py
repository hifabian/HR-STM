# @author Hillebrand, Fabian
# @date   2019


import numpy as np
import python.read_input as read_input
import python.util as util


# ==============================================================================
# ==============================================================================
class WFN:
  """!
    @brief A WFN-object for CP2K.

    As a WFN-object, it provides access to the wave function of a system
    as well as it's derivatives.

    @note This object uses MPI and divides states to different processes.
  """

  # ============================================================================
  
  def __init__(self, iFile, aFile, wFile, rank=0, size=1, comm=None):
    """!
      @param iFile CP2K input file used for SCF-calculation.
      @param aFile xyz file with atom positions.
      @param wFile File containing CP2K coefficients (.MOLog or .wfn).
      @param rank  Rank of process.
      @param size  Total number of processes.
      @param comm  MPI communication.
    """
    self.rank = rank
    self.size = size
    self.comm = comm

    # Rank 0 reads input files
    if rank == 0:
      self.abc = self._read_ABC(iFile)
      self.atoms = self._read_xyz(aFile, self.abc)
      coeffs = self._read_wfn(wFile)
      # TODO prepare for scattering of coefficients
      self.coeffs = coeffs
      
  # ============================================================================
  
  def _read_ABC(file):
    return read_input.read_ABC(file)
  def _read_xyz(file, abc=None):
    if abc is None:
      return read_input.read_xyz(file)
    return util.apply_box(read_input.read_xyz(file), abc)
  def _read_wfn(file):
      return read_input.read_wfn(file)

  # ============================================================================
  
    

# ==============================================================================
# ==============================================================================
