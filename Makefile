all: TOAT_PDOS_ROT 		 TOAT_WFN_ROT 		TOAT_PDOS 		TOAT_WFN \
	 TOAT_PDOS_BEADS_ROT TOAT_WFN_BEADS_ROT TOAT_PDOS_BEADS TOAT_WFN_BEADS

# ==============================================================================
# ==============================================================================

SAMPLE=examples/Samples
TIP=tips

OUTPUT=examples/plots

# ==============================================================================
# ==============================================================================

RUN=mpirun -np 1 python3 -m mpi4py run.py

# ==============================================================================
# ==============================================================================

TOAT_ARGS= $(RUN) \
	--cp2k_input $(SAMPLE)/TOAT/sample.inp \
	--basis_sets $(SAMPLE)/BASIS_MOLOPT \
	--xyz $(SAMPLE)/TOAT/sample.xyz \
	--coeffs $(SAMPLE)/TOAT/SAMPLE-RESTART.wfn \
	--tip_pos $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPpos \
			  $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPdisp \
	--tip_shift -3.0 \
	--orbs_tip 1 \
	--voltages -2.5 -1.6 -1.4 -1.2 -1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 2.5 \
	--hartree_file $(SAMPLE)/TOAT/SAMPLE-HART-v_hartree-1_0.cube

#TOAT_ARGS= $(RUN) \
	--cp2k_input_s $(SAMPLE)/TOAT/sample.inp \
	--basis_sets_s ../GitRepositories/cp2k/data/BASIS_MOLOPT \
	--xyz_s $(SAMPLE)/TOAT/sample.xyz \
	--coeffs_s $(SAMPLE)/TOAT/SAMPLE-RESTART.wfn \
	--pbc 1 1 0 \
	--tip_pos $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPpos \
			  $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPdisp \
	--tip_shift -3.0 \
	--heights 0.0 1.0 2.0 2.45 \
	--orbs_tip 1 \
	--voltages -2.5 -1.6 -1.4 -1.2 -1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 2.5 \
	--emin -5 --emax 5 --etip 0

# ==============================================================================
# ==============================================================================

PDOS= \
	--pdos_list $(TIP)/blunt/aiida-PDOS-list2-1.pdos \
				$(TIP)/blunt/aiida-PDOS-list1-1.pdos

# ==============================================================================
# ==============================================================================

TOAT_PDOS_ROT:
	$(TOAT_ARGS) $(PDOS) --output $(OUTPUT)/toat_tCO_pdos_rot --rotate

TOAT_PDOS:
	$(TOAT_ARGS) $(PDOS) --output $(OUTPUT)/toat_tCO_pdos

# ==============================================================================
# ==============================================================================

clean:
	rm -rf python/__pycache__ python/*/__pycache__
