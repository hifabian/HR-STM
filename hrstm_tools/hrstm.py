# @author Hillebrand, Fabian
# @date   2019

import numpy as np

import time

class Hrstm:
    """
    Provides a relatively genericc HR-STM simulator.

    Needs to be given an object for the wave function and the tip coefficients
    that provide certain information.

    This class supports parallelism. However, the grids should be divided along
    x-axis only.
    """

    def __init__(self, tip_coefficients, tip_grid_dim_all, sam_grid_matrix, fwhm, 
        mpi_rank=0, mpi_size=1, mpi_comm=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm
        self._tc = tip_coefficients
        self._tip_grid_dim_all = tip_grid_dim_all
        self._gm = sam_grid_matrix
        self._sigma = fwhm/2.35482


    def _dos(self, ene):
        """
        Gaussian density of states. 
        """
        return np.exp(-(ene / self._sigma)**2) / (self._sigma*(2*np.pi)**0.5)


    ### ------------------------------------------------------------------------
    ### Store and collect
    ### ------------------------------------------------------------------------

    def gather(self):
        """
        Gathers the current and returns it on rank 0.
        """
        if self.mpi_comm is None:
            return self.local_current
        if self.mpi_rank == 0:
            current = np.empty(self._tip_grid_dim_all+(len(self._voltages),))
        else:
            current = None
        outputs = self.mpi_comm.allgather(len(self.local_current.ravel()))
        self.mpi_comm.Gatherv(self.local_current, [current, outputs], root=0)
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
        current = self.gather()
        if self.mpi_rank == 0:
            for iheight in range(self._tip_grid_dim_all[-1]):
                for ivol in range(len(self._voltages)):
                    max_val = np.max(np.abs(current[:,:,iheight,ivol]))
                    current[:,:,iheight,ivol][np.abs(current[:,:,iheight,ivol]) < max_val*tol] = 0.0
            np.savez_compressed(filename, current.ravel())


    ### ------------------------------------------------------------------------
    ### Running HR-STM
    ### ------------------------------------------------------------------------

    def run(self, voltages):
        """!
        @brief Performs the HR-STM simulation.

        @param voltages List of bias voltages in eV.
        """
        self._voltages = np.array(voltages)

        # Currents for all bias voltages
        self.local_current = np.zeros((len(self._voltages),)+self._tc.grid_dim)

        # Over each separate tunnel process (e.g. to O- or C-atom)
        totTM = 0.0
        totVL = 0.0
        for itunnel in range(self._tc.ntunnels):
            for ispin_sam in range(self._gm.nspin):
                for iene_sam, ene_sam in enumerate(self._gm.ene[ispin_sam]):
                    for ispin_tip, enes_tip in enumerate(self._tc.ene):
                        ienes_tip = np.arange(len(enes_tip))
                        vals = (enes_tip*ene_sam > 0.0) \
                            | ((ene_sam <= 0.0) & (enes_tip == 0.0)) \
                            | ((ene_sam == 0.0) & (enes_tip <= 0.0))
                        skip = True
                        for voltage in self._voltages:
                            skip &= (np.abs(voltage-ene_sam+enes_tip) >= 4.0*self._sigma)
                        for iene_tip in [iene_tip for iene_tip in ienes_tip[~(skip | vals)]]:
                            ene_tip = self._tc.ene[ispin_tip][iene_tip]
                            start = time.time()
                            tunnel_matrix_squared = (np.einsum("i...,i...->...",
                                self._tc[itunnel,ispin_tip,iene_tip],
                                self._gm[itunnel,ispin_sam,iene_sam]))**2
                            end = time.time()
                            totTM += end-start
                            start = time.time()
                            for ivol, voltage in enumerate(self._voltages):
                                ene = voltage+ene_tip-ene_sam
                                if ((0 <= ene_tip <= -voltage) or (-voltage <= ene_tip <= 0)) \
                                    and abs(ene) < 4.0*self._sigma:
                                    self.local_current[ivol] += np.sign(ene_tip)*self._dos(ene) \
                                        * tunnel_matrix_squared
                            end = time.time()
                            totVL += end-start
        # Copy to assure C-contiguous array
        self.local_current = self.local_current.transpose((1,2,3,0)).copy()
        print("Total time for tunneling matrix was {:} seconds.".format(totTM))
        print("Total time for voltage loop was {:} seconds.".format(totVL))
