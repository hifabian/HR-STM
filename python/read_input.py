# @author Hillebrand, Fabian
# @date   2019

import numpy as np

# ------------------------------------------------------------------------------

hartreeToEV = 27.21138602

# ------------------------------------------------------------------------------

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


def read_PPPos(filename):
  """!
    @brief Loads the positions obtained from the probe particle model.

    In this case 'filename' refers to the head of files whose tail is assumed
    to be of the form '*_x.npy', '*_y.npy' and '*_z.npy.'
    
    The units are in Angstrom.

    @param filename Name of the start of the file without tail.

    @return List of positions in shape of mgrid.
            Matrix containing information on the unrelaxed grid.
  """
  disposX = np.transpose(np.load(filename+"_x.npy"))
  disposY = np.transpose(np.load(filename+"_y.npy"))
  disposZ = np.transpose(np.load(filename+"_z.npy"))
  # Information on the grid
  lvec = (np.load(filename+"_vec.npy"))
  # Stack arrays to form 4-dimensional array of size [noX, noY, noZ, 3]
  dispos = (disposX,disposY,disposZ)

  return dispos, lvec


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


def read_PDOS(filename, eMin=0.0, eMax=0.0):
  """!
    @brief Reads coefficients from *.pdos file and uses these to construct
           beta coefficients.

    The eigenvalues are shifted such that the Fermi energy is at 0 and 
    scaled such that the units are in eV.

    @param filename Name of *.pdos file.
    @param eMin     Minimal energy that is read in eV.
    @param eMax     Maximal energy that is read in eV.

    @return A list containing matrices. Rows correspond to eigenvalues
            while columns to orbitals.
            A list containing arrays for eigenvalues.
  """
  with open(filename) as f:
    lines = list(line for line in (l.strip() for l in f) if line)

    # TODO spins
    noSpins = 1

    noEigsTotal = len(lines)-2
    noDer = len(lines[1].split()[5:])

    homoEnergies = [float(lines[0].split()[-2])*hartreeToEV]

    pdos = [np.empty((noEigsTotal,noDer))]
    eigs = [np.empty((noEigsTotal))]

    # Read all coefficients, cut later
    for lineIdx, line in enumerate(lines[2:]):
      parts = line.split()
      eigs[0][lineIdx] = float(parts[1])*hartreeToEV
      pdos[0][lineIdx,:] = [float(val) for val in parts[3:]]

    # Cut coefficients to energy range
    startIdx = [None] * noSpins
    for spinIdx in range(noSpins):
      try:
        startIdx[spinIdx] = np.where(eigs[spinIdx] >= eMin+homoEnergies[spinIdx])[0][0]
      except:
        startIdx[spinIdx] = 0 
    endIdx = [None] * noSpins
    for spinIdx in range(noSpins):
      try:
        endIdx[spinIdx] = np.where(eigs[spinIdx] > eMax+homoEnergies[spinIdx])[0][0]
      except:
        endIdx[spinIdx] = len(eigs[spinIdx])
    if endIdx <= startIdx:
      raise ValueError("Restricted energy-range too restrictive: endIdx <= startIdx")

    eigs = [eigs[spinIdx][startIdx[spinIdx]:endIdx[spinIdx]] \
      - homoEnergies[spinIdx] for spinIdx in range(noSpins)]
    pdos = [pdos[spinIdx][startIdx[spinIdx]:endIdx[spinIdx],:] \
      for spinIdx in range(noSpins)]

  return pdos, eigs


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

