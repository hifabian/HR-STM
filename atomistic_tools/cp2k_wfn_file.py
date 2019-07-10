"""
Tools to put CP2K orbitals on a real space grid
""" 

import os
import numpy as np
import scipy
import scipy.io
import time
import copy
import sys

import gzip

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

class Cp2kWfnFile:
    """
    Class to deal with the CP2K .wfn file
    """
    
    def __init__(self, mpi_rank=0, mpi_size=1, mpi_comm=None):

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm

        # ------------------------------------------------------------------
        # All data directly from the .wfn file
        # See load_restart_wfn_file for descriptions
        self.natom = None
        self.nspin = None
        self.nao = None
        self.nset_max = None
        self.nshell_max = None

        self.nset_info = None
        self.nshell_info = None
        self.nso_info = None

        self.nmo = None
        self.lfomo = None
        self.nelectron = None

        self.i_homo_cp2k = None # cp2k homo indexes, takes also smearing into account (counting start from 1)
        self.evals = None
        self.occs = None

        self.i_homo = None # global indexes, corresponds to WFN nr (counting start from 1)
        self.homo_ens = None
        # ------------------------------------------------------------------
        # "selected data", meaning corresponding to energy/n_orb limits
        # (But still shared by the mpi processors)
        self.evals_sel = None
        self.occs_sel = None

        # ------------------------------------------------------------------
        # data additionally divided between mpi processors
        self.evals_loc = None
        self.coef_array = None # coef_array[ispin][i_mo, i_ao]
        self.i_homo_loc = None # indexes wrt to the first orbital of the MPI process

        # ------------------------------------------------------------------
        # Transformed into easily accessible format
        self.morb_composition = None # only for this mpi process
        self.glob_morb_energies = None # contains all energies in the selected range
        self.morb_energies = None # energies for this mpi process only
        self.ref_energy = None
        

    def write_ascii_gz(self, out_file):
        """
        Works only in serial mode!
        """
        if self.mpi_size != 1:
            print("Error: only serial calculation supported.")
            return False

        np.set_printoptions(threshold=np.inf)

        with gzip.open(out_file, 'wt') as f_out:
            f_out.write("%d %d %d %d %d\n" % (self.natom, self.nspin, self.nao, self.nset_max, self.nshell_max))

            f_out.write(np.array2string(self.nset_info) + "\n")
            f_out.write(np.array2string(self.nshell_info) + "\n")
            f_out.write(np.array2string(self.nso_info) + "\n")

            for ispin in range(self.nspin):

                if self.nspin == 1:
                    n_el = 2*(self.i_homo_loc[ispin]+1)
                else:
                    n_el = self.i_homo_loc[ispin]+1

                f_out.write("%d %d %d %d\n" % (len(self.coef_array[ispin]), self.i_homo_cp2k[ispin], self.lfomo[ispin], n_el))

                evals_occs = np.hstack([self.evals_sel[ispin], self.occs_sel[ispin]])
                f_out.write(np.array2string(evals_occs) + "\n")

                for imo in range(len(self.coef_array[ispin])):
                    f_out.write(np.array2string(self.coef_array[ispin][imo]) + "\n")

    def read_ascii_gz(self, in_file):
        """
        Works only in serial mode!
        """
        if self.mpi_size != 1:
            print("Error: only serial calculation supported.")
            return False

        def read_numpy_str(f_iter, dtype):
            lines = []
            while True:
                line = f_iter.readline()
                if line == "":
                    return None
                lines.append(line.strip()+" ")
                if lines[-1][-2] == ']':
                    break
            s = "".join(lines)
            return np.array(s[1:-2].split(), dtype=dtype)

        with gzip.open(in_file, 'rt') as f_in:
            self.natom, self.nspin, self.nao, self.nset_max, self.nshell_max = [ int(x) for x in f_in.readline().split() ]
            
            self.nset_info = read_numpy_str(f_in, int)
            self.nshell_info = read_numpy_str(f_in, int)
            self.nso_info = read_numpy_str(f_in, int)

            self.nmo = []
            self.lfomo = []
            self.nelectron = []
            self.i_homo_cp2k = []
            self.i_homo = []
            self.i_homo_loc = []

            self.evals = []
            self.occs = []

            self.coef_array = []
            self.homo_ens = []

            self.evals_sel = []
            self.occs_sel = []
            self.evals_loc = []

            for ispin in range(self.nspin):
                nmo_, i_homo_cp2k_, lfomo_, nelectron_ = [ int(x) for x in f_in.readline().split() ]

                if self.nspin == 1:
                    i_homo_ = int(nelectron_/2) - 1
                else:
                    i_homo_ = nelectron_ - 1

                self.nmo.append(nmo_)
                self.lfomo.append(lfomo_)
                self.nelectron.append(nelectron_)
                self.i_homo_cp2k.append(i_homo_cp2k_)
                self.i_homo.append(i_homo_)
                self.i_homo_loc.append(i_homo_)

                evals_occs = read_numpy_str(f_in, float)
                self.evals.append(evals_occs[:nmo_])
                self.occs.append(evals_occs[nmo_:])

                self.evals_loc.append(evals_occs[:nmo_])
                self.evals_sel.append(evals_occs[:nmo_])
                self.occs_sel.append(evals_occs[nmo_:])

                homo_en = self.evals[ispin][i_homo_]*hart_2_ev
                self.homo_ens.append(homo_en)

                self.coef_array.append([])
                for imo in range(nmo_):
                    self.coef_array[ispin].append(read_numpy_str(f_in, float))
    
    def write_fortran(self, out_file):
        """
        Works only in serial mode!
        """
        if self.mpi_size != 1:
            print("Error: only serial calculation supported.")
            return False

        outf = scipy.io.FortranFile(out_file, 'w')

        outf.write_record(np.array([self.natom, self.nspin, self.nao, self.nset_max, self.nshell_max], dtype=np.int32))

        outf.write_record(self.nset_info)
        outf.write_record(self.nshell_info)
        outf.write_record(self.nso_info)

        for ispin in range(self.nspin):

            if self.nspin == 1:
                n_el = 2*(self.i_homo_loc[ispin]+1)
            else:
                n_el = self.i_homo_loc[ispin]+1

            outf.write_record(np.array([len(self.coef_array[ispin]), self.i_homo_cp2k[ispin], self.lfomo[ispin], n_el], dtype=np.int32))

            evals_occs = np.hstack([self.evals_sel[ispin], self.occs_sel[ispin]])
            outf.write_record(evals_occs)

            for imo in range(len(self.coef_array[ispin])):
                outf.write_record(self.coef_array[ispin][imo])


    def load_restart_wfn_file(self, restart_file, emin=None, emax=None, n_homo=None, n_lumo=None):
        """ Reads the molecular orbitals from cp2k restart wavefunction file in specified energy range
        Note that the energy range is in eV and with respect to HOMO energy.
        
        sets member variables:
        * coef_array
        * ...
        """

        inpf = scipy.io.FortranFile(restart_file, 'r')

        self.natom, self.nspin, self.nao, self.nset_max, self.nshell_max = inpf.read_ints()
        # natom - number of atoms
        # nspin - number of spins
        # nao - number of atomic orbitals
        # nset_max - maximum number of sets in the basis set
        #           (e.g. if one atom's basis set contains 3 sets and every other
        #           atom's contains 1, then this value will still be 3)
        # nshell_max - maximum number of shells in each set

        # number of sets in the basis set for each atom
        self.nset_info = inpf.read_ints()

        # number of shells in each of the sets
        self.nshell_info = inpf.read_ints()

        # number of orbitals in each shell
        self.nso_info = inpf.read_ints()

        self.nmo = []
        self.lfomo = []
        self.nelectron = []
        self.evals = []
        self.occs = []

        # different HOMO indexes (for debugging and matching direct cube output)
        self.i_homo_cp2k = [] # cp2k homo indexes, takes also smearing into account (counting start from 1)
        self.i_homo = []      # global indexes, corresponds to WFN nr (counting start from 1)
        self.i_homo_loc = []  # indexes wrt to the first orbital of the MPI process

        self.homo_ens = []
        self.coef_array = []

        self.evals_sel = []
        self.occs_sel = []
        self.evals_loc = []

        for ispin in range(self.nspin):
            nmo_, i_homo_cp2k_, lfomo_, nelectron_ = inpf.read_ints()
            # nmo - number of molecular orbitals
            # homo - index of the HOMO
            # lfomo - ???
            # nelectron - number of electrons

            # Note that "i_homo_cp2k" is affected by smearing. to have the correct, T=0K homo:
            if self.nspin == 1:
                i_homo_ = int(nelectron_/2) - 1
            else:
                i_homo_ = nelectron_ - 1

            self.nmo.append(nmo_)
            self.lfomo.append(lfomo_)
            self.nelectron.append(nelectron_)

            self.i_homo_cp2k.append(i_homo_cp2k_)
            self.i_homo.append(i_homo_)

            # list containing all eigenvalues and occupancies of the molecular orbitals
            evals_occs = inpf.read_reals()
            self.evals.append(evals_occs[:nmo_])
            self.occs.append(evals_occs[nmo_:])
            
            evals = self.evals[ispin] * hart_2_ev
            homo_en = evals[i_homo_]
            self.homo_ens.append(homo_en)
            
            ### ---------------------------------------------------------------------
            ### Select orbitals in the specified range

            # NB: ind_start is inclusive and ind_end is exclusive
            ind_start = None
            ind_end = None

            # Energy range (if specified)
            if emin is not None and emax is not None:
                try:
                    ind_start = np.where(evals >= homo_en + emin)[0][0]
                except:
                    ind_start = 0
                try:
                    ind_end = np.where(evals > homo_en + emax)[0][0]
                except:
                    ind_end = len(evals)

                if evals[-1] < homo_en + emax:
                    print("WARNING: possibly not enough ADDED_MOS, last eigenvalue is %.2f" % (evals[-1]-homo_en))
            

            # num HOMO/LUMO range (if specified)
            if n_homo is not None:
                ind_start_n = i_homo_ - n_homo + 1
                if ind_start is None or ind_start_n < ind_start:
                    ind_start = ind_start_n
                if ind_start < 0:
                    print("WARNING: n_homo out of bounds.")
                    ind_start = 0
            if n_lumo is not None:
                ind_end_n = i_homo_ + n_lumo + 1
                if ind_end is None or ind_end_n > ind_end:
                    ind_end = ind_end_n
                if ind_end > len(evals):
                    print("WARNING: n_lumo out of bounds, increase ADDED_MOS.")
                    ind_end = len(evals)
            
            # If no limits are specified, take all orbitals
            if ind_start is None:
                ind_start = 0
            if ind_end is None:
                ind_end = len(evals)

            num_selected_orbs = ind_end - ind_start

            ### ---------------------------------------------------------------------
            ### Divide the orbitals between mpi processes

            # Select orbitals for the current mpi rank
            base_orb_per_rank = int(np.floor(num_selected_orbs/self.mpi_size))
            extra_orbs =  num_selected_orbs - base_orb_per_rank*self.mpi_size
            if self.mpi_rank < extra_orbs:
                loc_ind_start = self.mpi_rank*(base_orb_per_rank + 1) + ind_start
                loc_ind_end = (self.mpi_rank+1)*(base_orb_per_rank + 1) + ind_start
            else:
                loc_ind_start = self.mpi_rank*(base_orb_per_rank) + extra_orbs + ind_start
                loc_ind_end = (self.mpi_rank+1)*(base_orb_per_rank) + extra_orbs + ind_start

            print("R%d/%d, loading indexes %d:%d / %d:%d"%(self.mpi_rank, self.mpi_size,
                loc_ind_start, loc_ind_end-1, ind_start, ind_end-1))
            
            self.evals_sel.append(self.evals[ispin][ind_start:ind_end])
            self.occs_sel.append(self.occs[ispin][ind_start:ind_end])
            self.evals_loc.append(self.evals[ispin][loc_ind_start:loc_ind_end])

            ### ---------------------------------------------------------------------
            ### Read the coefficients from file

            self.coef_array.append([])

            first_imo = -1

            for imo in range(nmo_):
                coefs = inpf.read_reals()
                if imo < loc_ind_start:
                    continue
                if imo >= loc_ind_end:
                    if ispin == self.nspin - 1:
                        break
                    else:
                        continue
                
                if first_imo == -1:
                    first_imo = imo

                self.coef_array[ispin].append(coefs)
            
            self.coef_array[ispin] = np.array(self.coef_array[ispin])

            self.i_homo_loc.append(i_homo_ - first_imo) # Global homo index wrt to the initial MO
            ### ---------------------------------------------------------------------
            
        inpf.close()


    def convert_readable(self):
        """
        Intuitive indexing for the molecular orbital coefficients:
        * coef_array[ispin][i_mo, i_ao] -> morb_composition[ispin][iatom][iset][ishell][iorb][i_mo]

        Energies in eV wrt HOMO / avg of HOMOs
        * glob_morb_energies
        * morb_energies
        """
        self.morb_composition = []
        self.glob_morb_energies = []
        self.morb_energies = []

        ### ---------------------------------------------------------------------
        ### Intuitive indexing for the coefficients:
        ### coef_array[ispin][i_mo, i_ao] -> morb_composition[ispin][iatom][iset][ishell][iorb][i_mo]

        for ispin in range(self.nspin):

            self.morb_composition.append([]) # 1: spin index
            shell_offset = 0
            norb_offset = 0
            i_ao = 0
            for iatom in range(self.natom):
                nset = self.nset_info[iatom]
                self.morb_composition[ispin].append([]) # 2: atom index
                for iset in range(self.nset_max):
                    nshell = self.nshell_info[shell_offset]
                    shell_offset += 1
                    if nshell != 0:
                        self.morb_composition[ispin][iatom].append([]) # 3: set index
                    shell_norbs = []
                    for ishell in range(self.nshell_max):
                        norb = self.nso_info[norb_offset]
                        shell_norbs.append(norb)
                        norb_offset += 1
                        if norb == 0:
                            continue
                        self.morb_composition[ispin][iatom][iset].append([]) # 4: shell index (l)
                        for iorb in range(norb):
                            self.morb_composition[ispin][iatom][iset][ishell].append([]) # 5: orb index (m)
                            if self.coef_array[ispin].shape[0] != 0:
                                self.morb_composition[ispin][iatom][iset][ishell][iorb] = self.coef_array[ispin][:, i_ao] # 6: mo index
                            else:
                                self.morb_composition[ispin][iatom][iset][ishell][iorb] = np.array([])
                            # [iatom][iset][ishell][iorb] -> determine [i_ao]
                            i_ao += 1

        ### ---------------------------------------------------------------------
        ### Energies in eV wrt HOMO / avg of HOMOs

        if self.nspin == 1:
            self.ref_energy = self.homo_ens[0]
        else:
            self.ref_energy = 0.5*(self.homo_ens[0]+self.homo_ens[1])

        for ispin in range(self.nspin):
            self.glob_morb_energies.append(self.evals_sel[ispin]*hart_2_ev-self.ref_energy)
            self.morb_energies.append(self.evals_loc[ispin]*hart_2_ev-self.ref_energy)
        
