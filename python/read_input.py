# @author Hillebrand, Fabian
# @date   2018-2019

import numpy as np
import scipy.io

# ------------------------------------------------------------------------------

# Constants
hartreeToEV = 27.21138602

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

class Atom():
  def __init__(self, symbol, position):
    self._symbol = symbol
    self._position = position

  @property
  def symbol(self):
    return self._symbol
  @property
  def position(self):
    return self._position


class Atoms():
  def __init__(self, filename):
    with open(filename) as f:
      positions = []
      for lineIdx, line in enumerate(f):
        if lineIdx == 0:
          noAtoms = int(line.split()[0])
          self._atoms = [None]*noAtoms
          self._positions = np.empty((noAtoms,3))
          continue
        if lineIdx == 1:
          continue
        parts = line.split()
        self._atoms[lineIdx-2] = Atom(parts[0], np.array([float(val) for val in parts[1:4]]))
        self._positions[lineIdx-2] = np.array([float(val) for val in parts[1:4]])

  @property 
  def positions(self):
    return self._positions
  def get_positions(self):
    return self.positions
  def __getitem__(self, idx):
    return self._atoms[idx]
  def __len__(self):
    return len(self._atoms)


def read_xyz(filename):
  """!
    @brief Reads an *.xyz file.

    The units are assumed to be in Angstrom and will no be modified.

    @param filename Name of *.xyz file.

    @return A class containing the information from the *.xyz file.
  """
  return Atoms(filename)


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


def read_ABC(filename):
  """! @brief Reads the ABC from a CP2K input file. """
  with open(filename) as f:
    lines = f.readlines()
    for lineIdx in range(len(lines)):
      parts = lines[lineIdx].split()
      if len(parts) == 0:     # Skipping empty lines
        continue
      if parts[0] == "&CELL":
        abc = [float(val) for val in lines[lineIdx+1].split()[1:]]
        break
  return abc


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


def read_MO(filename, eMin=None, eMax=None):
  """!
    @brief Reads a *.MOLog file within a certain energy range. 

    This is made to be compatible with CP2K. The atomic orbitals are built from
    contracted Gaussians. See also: https://www.cp2k.org/basis_sets

    The energy range is given in eV and centered around the HOMO energy.

    The energy units are assumed to be in Hartree in the input file and will be
    converted to eV. Furthermore, the energies will be shifted to a reference 
    energy given by the HOMO energy.

    @attention The coefficients must be stored in the spherical basis.

    
    @param filename Name of *.MOLog file.
    @param eMin     Minimal energy that is read in eV (exclusive).
    @param eMax     Maximum energy that is read in eV (inclusive).

    @return A container for the basis function coefficients. To get a specific
            coefficient, one can use [ATOM][SPIN][BASIS,ENERGY] where:
              ATOM   - Index for atom.
              SPIN   - Index for spin (used in unrestricted Kohn-Sham).
              BASIS  - Index belonging to basis function on atom.
              ENERGY - Index for the energy.
            A container for the energies. To get a specific energy, one can use
            [SPIN][ENERGY] where:
              SPIN   - Index for spin (used in unrestricted Kohn-Sham).
              ENERGY - Index for the energy.
            The reference energy.
  """

  with open(filename) as f:
    # All lines without empty ones
    lines = list(line for line in (l.strip() for l in f) if line)

    # Asserting we have indeed a spherical basis
    assert "SPHERICAL" in lines[0]

    if "HOMO-LUMO" in lines[-1]:
      hlIdx = 1
    else:
      hlIdx = 0

    # Number of basis functions (for 1 spin)
    noBasis = int(lines[-2-hlIdx].split()[0])
    # Number of atoms
    noAtoms = int(lines[-2-hlIdx].split()[1])

    # Determine numbers of spin (1 for restricted, 2 for unrestricted)
    if "ALPHA" in lines[0]:
      noSpins = 2
      # Data start for Beta spin
      spinOffset = lines.index("BETA MO EIGENVALUES, MO OCCUPATION NUMBERS, AND SPHERICAL MO EIGENVECTORS")
      noBeta = int(lines[-noBasis-4-hlIdx].split()[-1])
      noAlpha = int(lines[-spinOffset-noBasis-4-hlIdx].split()[-1])
      noEigs = np.array([noAlpha, noBeta])
      # Eigenvalues (orbital energies)
      eigs = [np.empty(noEigs[0]), np.empty(noEigs[1])]
    else:
      noSpins = 1
      spinOffset = 0
      noEigs = np.array([int(lines[-noBasis-4-hlIdx].split()[-1])])
      # Eigenvalues (orbital energies)
      eigs = [np.empty(noEigs[0])]

    homoEnergies = [None] * noSpins
    if noSpins == 1:
      homoEnergies[0] = float(lines[-1-hlIdx].split()[-1]) * hartreeToEV
    else:
      homoEnergies[0] = float(lines[spinOffset-1-hlIdx].split()[-1]) * hartreeToEV
      homoEnergies[1] = float(lines[-1-hlIdx].split()[-1]) * hartreeToEV

    for spinIdx in range(noSpins):
      # Index for which eigenvalues are currently read
      currentEigsIdx = 0
      # Read eigenvalues in packs
      for packIdx in range((noEigs[spinIdx]+3) // 4):
        # Read current line
        parts = [float(val) for val in lines[spinOffset*spinIdx+2+packIdx*(noBasis+3)].split()]
        # Get length to determine how many eigenvalues are on it
        noEigsOnLine = len(parts)
        # Read values
        eigs[spinIdx][currentEigsIdx:currentEigsIdx+noEigsOnLine] = parts
        # Iterate index
        currentEigsIdx += noEigsOnLine

    for spinIdx in range(noSpins):
      eigs[spinIdx] *= hartreeToEV


    # Restrict energy range
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

    # Restrict also eigenvalues
    eigs = [eigs[spinIdx][startIdx[spinIdx]:endIdx[spinIdx]] for spinIdx in range(noSpins)]
    noEigs = [len(eigs[spinIdx]) for spinIdx in range(noSpins)]


    # Coefficients
    coeffs = [None] * noAtoms
    # Offset for atom basis
    atomOffset = 0
    for atomIdx in range(noAtoms):
      # Spin on each atom
      coeffs[atomIdx] = [None] * noSpins

      for spinIdx in range(noSpins):
        # Number of basis function on this atom        
        noBasisAtom = sum([int(line.split()[1]) == atomIdx+1 for line in lines[4:4+noBasis]])

        coeffs[atomIdx][spinIdx] = np.empty((noBasisAtom, endIdx[spinIdx]-startIdx[spinIdx]))

        # Index for which coefficients are currently read
        currentCoeffIdx = 0
        # Read coefficients in packs
        for packIdx in range((noEigs[spinIdx]+3) // 4):

          # Index of line where pack actually starts
          startOfPack = spinOffset*spinIdx+1+(packIdx+startIdx[spinIdx]//4)*(noBasis+3)

          headerLine = lines[startOfPack].split()
          noCoeffsOnLine = len(headerLine)

          # Get start and end indices of relevant energies
          start = 0
          end = noCoeffsOnLine
          if str(startIdx[spinIdx]+1) in headerLine:
            start = headerLine.index(str(startIdx[spinIdx]+1))
          if str(endIdx[spinIdx]+1) in headerLine:
            end = headerLine.index(str(endIdx[spinIdx]+1))

          for basisIdx in range(noBasisAtom):
            currentLine = lines[atomOffset+startOfPack+3+basisIdx]
            coeffs[atomIdx][spinIdx][basisIdx,currentCoeffIdx:currentCoeffIdx+end-start] = \
              [float(val) for val in currentLine.split()[4+start:4+end]]
          # Update the index for which coefficients are currently written
          currentCoeffIdx += end-start
      # Update the offset for the basis functions
      atomOffset += noBasisAtom
  
  # Shift eigenvalues such that HOMO is reference point (0.0)
  if noSpins == 1:
    referenceEnergy = homoEnergies[0]
  else:
    referenceEnergy = sum(homoEnergies) / 2.0
  for spinIdx in range(noSpins):
    eigs[spinIdx] -= referenceEnergy

  return coeffs, eigs, referenceEnergy


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def read_wfn(filename, eMin=None, eMax=None):
  """!
    @brief Reads a RESTART file (*.wfn) within a certain energy range.

    This is made to be compatible with CP2K. The atomic orbitals are built from
    contracted Gaussians. See also: https://www.cp2k.org/basis_sets

    The energy range is given in eV and centered around the HOMO energy.

    The energy units are assumed to be in Hartree in the input file and will be
    converted to eV. Furthermore, the energies will be shifted to a reference 
    energy given by the HOMO energy.

    @param filename Name of *.wfn file.
    @param eMin     Minimal energy that is read in eV (exclusive).
    @param eMax     Maximum energy that is read in eV (inclusive).

    @return A container for the basis function coefficients. To get a specific
            coefficient, one can use [ATOM][SPIN][BASIS,ENERGY] where:
              ATOM   - Index for atom.
              SPIN   - Index for spin (used in unrestricted Kohn-Sham).
              BASIS  - Index belonging to basis function on atom.
              ENERGY - Index for the energy.
            A container for the energies. To get a specific energy, one can use
            [SPIN][ENERGY] where:
              SPIN   - Index for spin (used in unrestricted Kohn-Sham).
              ENERGY - Index for the energy.
            The reference energy.
  """

  inpf = scipy.io.FortranFile(filename, 'r')

  # noAtom     Number of atoms.
  # noSpin     Number of spins.
  # noAo       Number of atomic orbitals.
  # noSetMax   Maximum number of sets in the basis set (e.g. if one atom's
  #            basis set contains 3 sets and every other atoms contain 1,
  #            then this value will still be 3).
  # noShellMax Maximum number of shells in each set.
  noAtoms, noSpins, noAos, noSetsMax, noShellsMax = inpf.read_ints()

  # Number of basis sets for each atom. Total length is: noAtoms
  noSetsInfo = inpf.read_ints()

  # Number of basis function in each set on each atom
  # Note: For every basis set storage of size noSetsMax is reserved (filled with 0)!
  #       Thus the total length is: noSetsMax * noAtoms
  noShellsInfo = inpf.read_ints()

  # Number of orbitals for l in each basis set on each atom
  # Note: For every basis set storage of size noShellsMax is reserved (filled with 0)!
  #       Thus the total length is: noShellsMax * noSetsMax * noAtoms
  noSosInfo = inpf.read_ints()


  # Number of basis functions on atom
  noBasis = [None] * noAtoms
  for atomIdx in range(noAtoms):
    noBasis[atomIdx] = sum(noSosInfo[atomIdx*noSetsMax*noShellsMax:(atomIdx+1)*noSetsMax*noShellsMax])


  # Coefficiens for basis functions
  coeffs = [None] * noAtoms
  for atomIdx in range(noAtoms):
    coeffs[atomIdx] = [None] * noSpins
  # Eigenvalues (energies)
  eigs = [None] * noSpins
  # Occupation numbers
  occs = [None] * noSpins

  # HOMO energies
  homoEnergies = [None] * noSpins


  for spinIdx in range(noSpins):
    # noMos       Number of molecular orbitals (total number of eigenvalues).
    # homoIdx     Index for the HOMO.
    # lfomo       ???
    # noElectrons Number of electrons.
    noMos, homoIdx, lfomo, noElectrons = inpf.read_ints()

    # Account HOMO index for spin
    if noSpins == 1:
      homoIdx = noElectrons // 2 - 1
    else: # FOTRAN indexes from 1 (Python (C++) does from 0)
      homoIdx = noElectrons - 1

    # Read eigenvalues and occupancies
    eigsAndOccs = inpf.read_reals()
    eigs[spinIdx] = eigsAndOccs[:len(eigsAndOccs) // 2]
    occs[spinIdx] = eigsAndOccs[len(eigsAndOccs) // 2:]

    # Hartree to eV
    eigs[spinIdx] *= hartreeToEV


    # Store HOMO energy for this spin
    homoEnergies[spinIdx] = eigs[spinIdx][homoIdx]

    # Restrict energy
    try:
      startIdx = np.where(eigs[spinIdx] >= eMin+homoEnergies[spinIdx])[0][0]
    except:
      startIdx = 0
    try:
      endIdx = np.where(eigs[spinIdx] > eMax+homoEnergies[spinIdx])[0][0]-1
    except:
      endIdx = len(eigs[spinIdx])-1
    noEigs = endIdx-startIdx+1
    if endIdx < startIdx:
      raise ValueError("Restricted energy-range too restrictive: endIdx <= startIdx")
    eigs[spinIdx] = np.array(eigs[spinIdx][startIdx:endIdx+1])
    occs[spinIdx] = np.array(occs[spinIdx][startIdx:endIdx+1], dtype=int)


    # Build coefficient container
    for atomIdx in range(noAtoms):
      coeffs[atomIdx][spinIdx] = np.zeros((noBasis[atomIdx], noEigs))


    # Index for eigenvalues (restricted moIdx)
    eneIdx = 0
    for moIdx in range(noMos):
      coeffsMo = inpf.read_reals()


      # Check if in energy range
      if moIdx < startIdx:
        continue
      if moIdx > endIdx:
        if spinIdx == noSpins-1:
          break
        else:
          continue

      orbOffset = 0
      for atomIdx in range(noAtoms):
        for basisIdx in range(noBasis[atomIdx]):
          coeffs[atomIdx][spinIdx][basisIdx, eneIdx] = coeffsMo[orbOffset]
          # Increment orbital offset
          orbOffset += 1
      # Increment index for eigenvalues
      eneIdx += 1

  # Close fortran file
  inpf.close()

  # Shift eigenvalues such that HOMO is reference point (0.0)
  if noSpins == 1:
    referenceEnergy = homoEnergies[0]
  else:
    referenceEnergy = sum(homoEnergies) / 2.0
  for spinIdx in range(noSpins):
    eigs[spinIdx] -= referenceEnergy

  return coeffs, eigs, referenceEnergy


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

    @return A 4-dimensional array corresponding to (z,y,x,3).
  """
  disposX = np.transpose(np.load(filename+"_x.npy"))
  disposY = np.transpose(np.load(filename+"_y.npy"))
  disposZ = np.transpose(np.load(filename+"_z.npy"))
  # Information on the grid
  lvec = (np.load(filename+"_vec.npy"))
  # Stack arrays to form 4-dimensional array of size [noX, noY, noZ, 3]
  dispos = np.stack((disposX, disposY, disposZ), axis=3).copy()

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
