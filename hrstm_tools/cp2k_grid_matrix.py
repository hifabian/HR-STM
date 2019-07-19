# @author Hillebrand, Fabian
# @date   2018-2019

import numpy as np

from .interpolator import Interpolator

ang2bohr   = 1.88972612546

class Cp2kGridMatrix:
    """
    Class that provides a wrapper for Cp2kGridOrbtials such that they can be
    evaluated on an arbitrary grid using interpolation. Derivatives are also
    made accessible.

    This structure provides access to the wave functions via bracket operators.
    The following structure is provided: [itunnel,ispin,iene][derivative,x,y,z].
    Note that this evaluation may be performed lazily.
    """

    def __init__(self, cp2k_grid_orb, eval_region, tip_pos, norbs_tip, 
        mpi_rank=0, mpi_size=1, mpi_comm=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm
        self.norbs_tip = norbs_tip
        self.nspin = cp2k_grid_orb.nspin
        self.ase_atoms = cp2k_grid_orb.ase_atoms
        self._grids = tip_pos
        self._ene = []
        self._wfn_matrix = []
        self.dv = cp2k_grid_orb.dv / ang2bohr
        self._wfn = None

        # TODO separate this away because this way mpi_comm cannot be None
        # Distribute and gather energies and wave functions on MPI ranks
        for ispin in range(self.nspin):
            # Gather energies
            ene_separated = self.mpi_comm.allgather(
                cp2k_grid_orb.morb_energies[ispin])
            nene_by_rank = np.array([len(val) for val in ene_separated])
            self.ene.append(np.hstack(ene_separated))
            # Indices needed for the tip position on MPI rank
            isx = np.array([\
                np.floor(min([np.min(pos[0]-eval_region[0][0]) 
                              for pos in tip_pos[1:]]) / self.dv[0]),
                 np.ceil(max([np.max(pos[0]-eval_region[0][0]) 
                              for pos in tip_pos[1:]]) / self.dv[0]),
                ], dtype=int)
            isx_all = self.mpi_comm.allgather(isx)
            # Dimension of local grid for wave function matrix
            wfn_dim_local = (isx[1]-isx[0]+1,)+np.shape(
                cp2k_grid_orb.morb_grids[ispin])[2:]
            npoints = np.product(wfn_dim_local)
            # Gather the necessary stuff
            for rank in range(self.mpi_size):
                if self.mpi_rank == rank:
                    recvbuf = np.empty(len(self.ene[ispin])*npoints, 
                        dtype=cp2k_grid_orb.morb_grids[ispin].dtype)
                else:
                    recvbuf = None
                sendbuf = cp2k_grid_orb.morb_grids[ispin]\
                    [:,isx_all[rank][0]:isx_all[rank][1]+1].ravel()
                self.mpi_comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf,
                    nene_by_rank*npoints], root=rank)
                if self.mpi_rank == rank:
                    self._wfn_matrix.append(recvbuf.reshape(
                        (len(self.ene[ispin]),) + wfn_dim_local))

        # Set evaluation region for this MPI rank
        self.eval_region_local = eval_region
        self.eval_region_local[0] = eval_region[0][0]+isx*self.dv[0]
        self.wfn_dim = np.shape(self._wfn_matrix[0])[1:]
        self._reg_grid = ( \
            np.linspace(self.eval_region_local[0][0],
                        self.eval_region_local[0][1],self.wfn_dim[0]),
            np.linspace(self.eval_region_local[1][0],
                        self.eval_region_local[1][1],self.wfn_dim[1]),
            np.linspace(self.eval_region_local[2][0],
                        self.eval_region_local[2][1],self.wfn_dim[2]))


    ### ------------------------------------------------------------------------
    ### Access operators
    ### ------------------------------------------------------------------------

    @property
    def ene(self):
        return self._ene
    @property
    def grid_dim(self):
        return np.shape(self._grids[0])[1:]

    def __getitem__(self, itupel):
        igrid, ispin, iene = itupel
        # Check if already evaluated
        if self._wfn is not None \
            and self._cgrid == igrid \
            and self._cspin == ispin \
            and self._cene == iene:
            return self._wfn
        # Storage container
        self._wfn = np.empty(((self.norbs_tip+1)**2,)+self.grid_dim)

        # Create interpolator
        interp = Interpolator(self._reg_grid, self._wfn_matrix[ispin][iene])
        self._wfn[0] = interp(*self._grids[igrid])
        if self.norbs_tip:
            self._wfn[1] = interp.gradient(*self._grids[igrid],1)
            self._wfn[2] = interp.gradient(*self._grids[igrid],2)
            self._wfn[3] = interp.gradient(*self._grids[igrid],3)
        self._cgrid, self._cspin, self._cene = itupel
        return self._wfn
