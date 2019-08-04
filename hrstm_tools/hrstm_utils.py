# @author Hillebrand, Fabian
# @date   2019

import numpy as np
from mpi4py import MPI

hartreeToEV = 27.21138602


### ----------------------------------------------------------------------------
### Old read functions
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
    disposX = np.transpose(np.load(filename+"_x.npy")).copy()
    disposY = np.transpose(np.load(filename+"_y.npy")).copy()
    disposZ = np.transpose(np.load(filename+"_z.npy")).copy()
    # Information on the grid
    lvec = (np.load(filename+"_vec.npy"))
    # Stack arrays to form 4-dimensional array of size [3, noX, noY, noZ]
    dispos = (disposX,disposY,disposZ)

    return dispos, lvec


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


### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------


def apply_bounds(grid, lVec):
    """!
    @brief Restrict grids to box in x- and y-direction.

    @param grid Grid to be restricted.
    @param lVec Information on box.

    @return Grid with position restricted to periodic box.
    """
    dx = lVec[1,0]-lVec[0,0]
    grid[0][grid[0] >= lVec[1,0]] -= dx
    grid[0][grid[0] < lVec[0,0]]  += dx

    dy = lVec[2,1]-lVec[0,1]
    grid[1][grid[1] >= lVec[2,1]] -= dy
    grid[1][grid[1] < lVec[0,1]]  += dy

    return grid


def read_tip_positions(files, shift, dx, mpi_rank=0, mpi_size=1, mpi_comm=None):
    """
    Function to read tip positions and to determine grid orbital evaluation 
    region for sample (determined using tip positions).

    @return pos             List with positions for this rank.
            grid_dim        Total dimension of positions.
            sam_eval_region Region encompassing all positions.
            lVec            Region of non-relaxed positions (Oxygen).
    """
    # Only reading on one rank, could be optimized but not the bottleneck
    if mpi_rank == 0:
        pos_all = []
        for filename in files:
            positions, lVec = read_PPPos(filename)
            pos_all.append(positions)
        grid_dim = np.shape(pos_all[0])[1:]
        # Metal tip (needed only for rotation, no tunnelling considered)
        pos_all.insert(0, np.mgrid[ \
            lVec[0,0]:lVec[0,0]+lVec[1,0]:grid_dim[0]*1j, \
            lVec[0,1]:lVec[0,1]+lVec[2,1]:grid_dim[1]*1j, \
            lVec[0,2]-shift:lVec[0,2]+lVec[3,2]-shift \
            :grid_dim[2]*1j])
        # Evaluation region for sample (x,y periodic)
        xmin = lVec[0,0]
        xmax = lVec[0,0]+lVec[1,0]
        ymin = lVec[0,1]
        ymax = lVec[0,1]+lVec[2,1]
        zmin = min([np.min(pos[2]) for pos in pos_all[1:]])-dx/2
        zmax = max([np.max(pos[2]) for pos in pos_all[1:]])+dx/2
        sam_eval_region = np.array([[xmin,xmax], [ymin,ymax], [zmin,zmax]])
        # No MPI
        if mpi_comm is None:
            return pos_all, grid_dim, sam_eval_region, lVec
    else:
        pos_all = [[None]*3]*(len(files)+1)
        lVec = None
        grid_dim = None
        sam_eval_region = None
    # Broadcast small things
    lVec = mpi_comm.bcast(lVec, root=0)
    grid_dim = mpi_comm.bcast(grid_dim, root=0)
    sam_eval_region = mpi_comm.bcast(sam_eval_region, root=0)
    # Divide up tip positions along x-axis
    all_x_ids = np.array_split(np.arange(grid_dim[0]), mpi_size)
    lengths = [len(all_x_ids[rank])*np.product(grid_dim[1:]) 
        for rank in range(mpi_size)]
    offsets = [all_x_ids[rank][0]*np.product(grid_dim[1:]) 
        for rank in range(mpi_size)]
    # Prepare storage and then scatter grids
    pos = [np.empty((3,len(all_x_ids[mpi_rank]),)+grid_dim[1:])
        for i in range(len(files)+1)]
    for gridIdx in range(len(files)+1):
        for axis in range(3):
            mpi_comm.Scatterv([pos_all[gridIdx][axis], lengths, offsets, 
                MPI.DOUBLE], pos[gridIdx][axis], root=0)
    return pos, grid_dim, sam_eval_region, lVec


def create_tip_positions(eval_region, dx, mpi_rank=0, mpi_size=1, mpi_comm=None):
    """
    Creates uniform grids for tip positions. Due to the structure of the code,
    this returns a tuple with twice the same grid. Rotations are not supported.

    @return pos             List with positions for this rank.
            grid_dim        Total dimension of positions.
            sam_eval_region Region encompassing all positions.
            lVec            Region of non-relaxed positions (Oxygen).
    """
    eval_region = np.reshape(eval_region,(3,2))
    lVec = np.zeros((4,3))
    lVec[0,0] = eval_region[0,0]
    lVec[1,0] = eval_region[0,1]-eval_region[0,0]
    lVec[0,1] = eval_region[1,0]
    lVec[2,1] = eval_region[1,1]-eval_region[1,0]
    lVec[0,2] = eval_region[2,0]
    lVec[3,2] = eval_region[2,1]-eval_region[2,0] 
    grid_dim = (int(lVec[1,0]/dx+1), int(lVec[2,1]/dx+1), int(lVec[3,2]/dx+1))
    # True spacing
    dxyz = [lVec[i+1,i] / (grid_dim[i]-1) for i in range(3)]
    all_x_ids = np.array_split(np.arange(grid_dim[0]), mpi_size)
    start = lVec[0,0]+dxyz[0]*all_x_ids[mpi_rank][0]
    end = lVec[0,0]+dxyz[0]*all_x_ids[mpi_rank][-1]
    grid = np.mgrid[ \
        start:end:len(all_x_ids[mpi_rank])*1j,
        lVec[0,1]:lVec[0,1]+lVec[2,1]:grid_dim[1]*1j,
        lVec[0,2]:lVec[0,2]+lVec[3,2]:grid_dim[2]*1j]
    # Increase eval_region to account for non-periodic axis
    eval_region[2,0] -= dxyz[-1] / 2
    eval_region[2,1] += dxyz[-1] / 2
    # Positions emulating apex atom + metal tip
    pos = [grid, grid]
    return pos, grid_dim, eval_region, lVec
