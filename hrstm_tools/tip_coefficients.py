# @author Hillebrand, Fabian
# @date   2019

import numpy as np
import scipy as sp
import math

import time

from .hrstm_utils import read_PDOS


def const_coeffs(emin, emax, de=0.1, s=0.0, py=0.0, pz=0.0, px=0.0):
    """
    Creates coefficients for seperated tunnelling to each orbital.
    """
    nE = int((emax-emin) / de)+1
    cc = np.array([s, py, pz, px]) != 0.0
    coeffs = np.empty((nE*sum(cc),4),)
    ene = np.empty(nE*sum(cc))
    for n in range(nE):
        idx = n*(sum(cc))
        if s != 0.0:
            coeffs[idx+sum(cc[:1])-1,:] = [s**0.5, 0.0, 0.0, 0.0]
        if py != 0.0:
            coeffs[idx+sum(cc[:2])-1,:] = [0.0, py**0.5, 0.0, 0.0]
        if pz != 0.0:
            coeffs[idx+sum(cc[:3])-1,:] = [0.0, 0.0, pz**0.5, 0.0]
        if px != 0.0:
            coeffs[idx+sum(cc[:4])-1,:] = [0.0, 0.0, 0.0, px**0.5]
        ene[idx:idx+sum(cc)] = eMin+de*n

    return [coeffs], [ene]


class TipCoefficients:
    """
    Structure that provides access to tip coefficients for each point in a list
    of grids. 

    The coefficients can be rotated for the grid. The rotation is given through
    two different grids corresponding to the points of rotation and the rotated
    points. The axis of rotation is fixed in the x-y-plane.

    This structure provides access to the coefficients via bracket operators.
    The following structure is provided: [itunnel,ispin,iene][iorb,x,y,z].
    Note that this evaluation may be performed lazily.
    """

    def __init__(self, mpi_rank=0, mpi_size=1, mpi_comm=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm


    ### ------------------------------------------------------------------------
    ### Read function for coefficients
    ### ------------------------------------------------------------------------

    def read_coefficients(self, norbs, pdos_list, emin, emax):
        """
        Reads coefficients from files or via command line if given in the
        shape (s, py, pz, px, de).
        Coefficients are broadcasted to all MPI processes.
        """
        self.norbs = norbs
        self._coeffs = None
        if self.mpi_rank == 0:
            self._singles = []
            idx = 0 # Index of input argument
            while idx < len(pdos_list):
                try:
                    single, self._ene = const_coeffs(emin, emax,
                        s=float(pdos_list[idx]),
                        py=float(pdos_list[idx+1]),
                        pz=float(pdos_list[idx+2]),
                        px=float(pdos_list[idx+3]),
                        de=float(pdos_list[idx+4]))
                    idx += 5
                except ValueError:
                    single, self._ene = read_PDOS(pdos_list[idx],
                        emin, emax)
                    # Take square root to obtain proper coefficients
                    for ispin in range(len(single)):
                        single[ispin] = \
                            single[ispin][:,:(self.norbs+1)**2]**0.5
                    idx += 1
                self._singles.append(single)
        else:
          self._singles = None
          self._ene = None
        # Broadcast untransformed coefficients and energies
        if self.mpi_comm is not None:
          self._singles = self.mpi_comm.bcast(self._singles, root=0)
          self._ene = self.mpi_comm.bcast(self._ene, root=0)


    ### ------------------------------------------------------------------------
    ### Initialize coefficients
    ### ------------------------------------------------------------------------

    def initialize(self, pos, rotate=False):
        """ 
        Computes rotational matrix if necessary.
        This method does not communicate. Positions should be split up 
        before hand to avoid unnecessary calculations.
        """
        self._grid_dim = np.shape(pos[0])[1:]
        self.ntunnels = len(pos)-1
        # s-orbtial on tip only
        if self.norbs == 0:
            return

        start = time.time()
        # p-orbital on tip
        self._rot_matrix = [None]*self.ntunnels
        if not rotate:
            for itunnel in range(self.ntunnels):
                self._rot_matrix[itunnel] = 1
        else:
            npoints = np.prod(self.grid_dim)
            shifted_pos = []
            rm_data = []
            for i in range(self.ntunnels):
                shifted_pos.append(np.array([
                    pos[i+1][0]-pos[i][0],
                    pos[i+1][1]-pos[i][1],
                    pos[i+1][2]-pos[i][2]]
                  ).reshape(3,npoints))
                rm_data.append(np.empty(9*npoints))
            # Sparse matrix storage as COO
            ihelper = np.arange(3*npoints,dtype=int)
            # rows = [0,1,2,...,0,1,2,...,0,1,2,...]
            rm_rows = np.tile(ihelper,3)
            # cols = [0,0,0,1,1,1,2,2,2,...]
            rm_cols = np.repeat(ihelper,3)
            del ihelper
            # Matrix data
            for itunnel in range(self.ntunnels):
                v = np.array([0.0,0.0,-1.0])
                # Rotated vector
                w = shifted_pos[itunnel]
                w /= np.linalg.norm(w,axis=0)
                # Rotation axis (no rotation around x-y)
                n = np.cross(v,w,axisb=0).transpose()
                # Trigonometric values
                cosa = np.dot(v,w)
                sina = (1-cosa**2)**0.5

                #  R = [[n[0]**2*(1-cosa)+cosa, n[0]*n[1]*(1-cosa)-n[2]*sina, n[0]*n[2]*(1-cosa)+n[1]*sina],
                #       [n[1]*n[0]*(1-cosa)+n[2]*sina, n[1]**2*(1-cosa)+cosa, n[1]*n[2]*(1-cosa)-n[0]*sina],
                #       [n[2]*n[1]*(1-cosa)-n[1]*sina, n[2]*n[1]*(1-cosa)+n[0]*sina, n[2]**2*(1-cosa)+cosa]]
                # Permutation matrix for betas: (y,z,x) -> (x,y,z)
                #  P = [[0,0,1],
                #       [1,0,0],
                #       [0,1,0]]
                # Note: The rotational matrix R is with respect to the sample which is R^T for the tip
                #       --> R to R^T for going from tip to sample, R^T to R for gradient
                rm_data[itunnel][:npoints]            = n[1]**2*(1-cosa)+cosa
                rm_data[itunnel][npoints:2*npoints]   = n[2]*n[1]*(1-cosa)-n[0]*sina
                rm_data[itunnel][2*npoints:3*npoints] = n[0]*n[1]*(1-cosa)+n[2]*sina
                rm_data[itunnel][3*npoints:4*npoints] = n[1]*n[2]*(1-cosa)+n[0]*sina
                rm_data[itunnel][4*npoints:5*npoints] = n[2]**2*(1-cosa)+cosa
                rm_data[itunnel][5*npoints:6*npoints] = n[0]*n[2]*(1-cosa)-n[1]*sina
                rm_data[itunnel][6*npoints:7*npoints] = n[1]*n[0]*(1-cosa)-n[2]*sina
                rm_data[itunnel][7*npoints:8*npoints] = n[2]*n[0]*(1-cosa)+n[1]*sina
                rm_data[itunnel][8*npoints:9*npoints] = n[0]**2*(1-cosa)+cosa

            # Build large matrices
            for itunnel in range(self.ntunnels):
                self._rot_matrix[itunnel] = \
                    sp.sparse.csr_matrix(sp.sparse.coo_matrix((rm_data[itunnel],
                    (rm_rows,rm_cols)), shape=(3*npoints, 3*npoints)))

        end = time.time()
        print("Rotational matrices took {} seconds".format(end-start))


    ### ------------------------------------------------------------------------
    ### Access operators
    ### ------------------------------------------------------------------------

    @property
    def ene(self):
        return self._ene
    @property
    def grid_dim(self):
      return self._grid_dim

    def __getitem__(self, ituple):
        """ @param ituple Index tuple (itunnel,ispin,iene) """
        # Unpack indices
        itunnel, ispin, iene = ituple
        if self._coeffs is not None \
            and self._cene == iene \
            and self._cspin == ispin \
            and self._ctunnel == itunnel:
            return self._coeffs
        # Storage container
        self._coeffs = np.empty(((self.norbs+1)**2,)+self.grid_dim)

        # s-orbitals: Are never rotated
        self._coeffs[0].fill(self._singles[itunnel][ispin][iene,0])
        if self.norbs > 0:
            # p-orbitals: Are rotated like vectors
            self._coeffs[1].fill(self._singles[itunnel][ispin][iene,1])
            self._coeffs[2].fill(self._singles[itunnel][ispin][iene,2])
            self._coeffs[3].fill(self._singles[itunnel][ispin][iene,3])
            # Flat view of coeffs[1:4]
            flat_coeffs = self._coeffs[1:4].ravel()
            # Provoke write into flat view instead of overwriting variable with [:]
            flat_coeffs[:] = self._rot_matrix[itunnel]*self._coeffs[1:4].flatten()
        # Save some information
        self._ctunnel, self._cspin, self._cene = ituple
        return self._coeffs
