# @author Hillebrand, Fabian
# @date   2019

import time

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

  def __init__(self, chenCoeffs, wfn, fwhm, rank, size, comm, dimGrid):
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
    self._dimGrid = dimGrid

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
    if self._rank == 0:
      current = np.empty(self._dimGrid+(len(self._voltages),))
    else:
      current = None
    outputSizes = self._comm.allgather(len(self.localCurrent.ravel()))
    self._comm.Gatherv(self.localCurrent, [current, outputSizes], root=0)
    return current

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
    self.localCurrent = np.zeros((len(self._voltages),)+dimGrid, \
      dtype=np.float64)

    # Over each separate tunnel process (e.g. to O- or C-atom)
    totTM = 0.0
    totVL = 0.0
    for tunnelIdx in range(self._chenCoeffs.noTunnels):
      for spinSamIdx in range(self._wfn.noSpins):
        for esIdx, eigSample in enumerate(self._wfn.eigs[spinSamIdx]):
          for spinTipIdx, eigsTip in enumerate(self._chenCoeffs.eigs):
            etIds = np.arange(len(eigsTip))
            vals = (eigsTip*eigSample > 0.0) \
              | ((eigSample <= 0.0) & (eigsTip == 0.0)) \
              | ((eigSample == 0.0) & (eigsTip <= 0.0))
            skip = True
            for voltage in self._voltages:
              skip &= (np.abs(voltage-eigSample+eigsTip) >= 4.0*self._sigma)
            for etIdx in [etIdx for etIdx in etIds[~(skip | vals)]]:
              eigTip = self._chenCoeffs.eigs[spinTipIdx][etIdx]
              start = time.time()
              tunnelMatrixSquared = (self._tunnelMatrix[tunnelIdx,spinTipIdx,etIdx, \
                spinSamIdx,esIdx])**2
              end = time.time()
              totTM += end-start
              start = time.time()
              for volIdx, voltage in enumerate(self._voltages):
                ene = voltage+eigTip-eigSample
                if abs(ene) < 4.0*self._sigma:
                  self.localCurrent[volIdx] += np.sign(eigTip)*self._dos(ene) \
                    * tunnelMatrixSquared
              end = time.time()
              totVL += end-start
    # Copy to assure C-contiguous array
    self.localCurrent = self.localCurrent.transpose((1,2,3,0)).copy()
    print("Total time for tunneling matrix was {:} seconds.".format(totTM))
    print("Total time for voltage loop was {:} seconds.".format(totVL))

################################################################################
