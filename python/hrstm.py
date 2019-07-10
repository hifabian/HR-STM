# @author Hillebrand, Fabian
# @date   2019

import numpy as np

# Generic interpolator that provides a gradient operation.
from interpolator import Interpolator

################################################################################
class HRSTM:
  """!
    @brief High-Resolution Scanning Microscopy Class.

    Combines tip coefficients and wave function to obtain the tunnelling 
    current. Tip coefficients and wave function are externally passed.

    Class may be called in parallel.

    @note If used in parallel, it is assumed that the grid is divided up as an
          array along the x-direction only.
  """

  def __init__(self):
    pass

  ##############################################################################
  def _compute(self):
    """!
      @brief Evaluates the current on the different processes. The current is
             stored in self.localCurrent and needs to be assembled.
    """
    pass

  ##############################################################################
  def gather(self):
    """!
      @brief Gathers the current and returns it on rank 0.
    """
    pass

  def write(self, filename):
    """!
      @brief Writes the current to a file (*.npy).

      The file is written as a 1-dimensional array. The reconstruction has
      thus be done by hand. It can be reshaped into a 4-dimensional array in
      the form [zIdx,yIdx,xIdx,vIdx].

      @param filename Name of file.
    """
    pass

  def write_compressed(self, filename, tol=1e-3):
    """!
      @brief Writes the current compressed to a file (*.npz).

      The file is written as a 1-dimensional array similar to write().
      Furthermore, in order to load the current use np.load()['arr_0'].

      @attention This method evokes a gather!

      @param filename Name of file.
      @param tol      Relative toleranz to the maximum for a height 
                      and voltage.
    """
    pass

  ##############################################################################
  def run(self, voltages):
    """!
      @brief Performs the HR-STM simulation.

      @param voltages List of bias voltages in eV.
    """
    pass

################################################################################
