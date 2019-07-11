# @author Hillebrand, Fabian
# @date   2019

import numpy as np

from basis.wavefunction_abc import *
from tunnelling.chen_coeffs_abc import *
from tunnelling.tunnel_matrix import *

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

  def __init__(self, chenCoeffs, wfn, fwhm, rank, size, comm):
    """!
      @brief Initiator

      @param chenCoeffs Chen coefficients.
      @param wfn        Wave function.
      @param fwhm       Full width at half maximum for sample density of states.
      @param rank       Rank of MPI process.
      @param size       Total number of MPI processes.
      @param comm       Communicator of MPI processes.
    """
    self._chenCoeffs = chenCoeffs
    self._wfn = wfn
    self._sigma = fwhm/2.35482
    self._rank = rank
    self._size = size
    self._comm = comm

  ##############################################################################
  def _compute(self):
    """!
      @brief Evaluates the current on the different processes. The current is
             stored in self.localCurrent and needs to be assembled.
    """
    pass

  def _dos(self, ene):
    """! @brief Gaussian density of states. """
    return np.exp(-(ene / self._sigma)**2) / (self._sigma*(2*np.pi)**0.5)
    

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
    self._voltages = voltages

    # Tunnel matrix object
    self._tunnelMatrix = TunnelMatrix(self._chenCoeffs, self._wfn)

    dimGrid = self._tunnelMatrix.dimGrid
    # Currents for all bias voltages
    self.localCurrent = np.zeros(dimGrid+(len(self._voltages),), \
      dtype=np.float64)

    # Over each separate tunnel process (e.g. to O- or C-atom)
    for tunnelIdx in range(self._chenCoeffs.noTunnels):
      for spinTipIdx in range(self._chenCoeffs.noSpins):
        for etIdx in range(self._chenCoeffs.noEigs[spinTipIdx]):
          eigTip = self._chenCoeffs.eigs[spinTipIdx][etIdx]
          for spinSamIdx in range(self._wfn.noSpins):
            for esIdx in range(self._wfn.noEigs[spinSamIdx]):
              eigSample = self._wfn.eigs[spinSamIdx][esIdx]
              # Only if tip and sample are occupied/unoccupied
              if eigTip*eigSample > 0.0 \
                or (eigTip == 0.0 and eigSample <= 0.0) \
                or (eigSample == 0.0 and eigTip <= 0.0):
                continue

              # Checking if density of states will be 0 anyway
              skip = True
              for voltage in self._voltages:
                ene = voltage+eigTip-eigSample
                if abs(ene) < 4.0*self._sigma:
                  skip = False
                  break
              if skip:
                continue

              # Otherwise compute tunnelling matrix entry
              tunnelMatrixSquared = (self._tunnelMatrix[tunnelIdx,spinTipIdx,etIdx, \
                spinSamIdx,esIdx])**2
              for volIdx, voltage in enumerate(self._voltages):
                ene = voltage+eigTip-eigSample
                if  abs(ene) >= 4.0*self._sigma:
                  continue
                self.localCurrent[...,volIdx] += np.sign(eigTip)*self._dos(ene) \
                  * tunnelMatrixSquared

################################################################################
