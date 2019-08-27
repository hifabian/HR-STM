all: TOAT_PDOS_ROT 		 TOAT_WFN_ROT 		TOAT_PDOS 		TOAT_WFN \
	 TOAT_PDOS_BEADS_ROT TOAT_WFN_BEADS_ROT TOAT_PDOS_BEADS TOAT_WFN_BEADS

# ==============================================================================
# ==============================================================================

SAMPLE=examples/Samples
TIP=tips

OUTPUT=examples/plots

# ==============================================================================
# ==============================================================================

RUN=mpirun -np 3 python3 -m mpi4py run.py

# ==============================================================================
# ==============================================================================

TOAT_ARGS= $(RUN) \
	--cp2k_input_file $(SAMPLE)/TOAT/sample.inp \
	--basis_set_file $(SAMPLE)/BASIS_SET \
	--xyz_file $(SAMPLE)/TOAT/sample.xyz \
	--wfn_file $(SAMPLE)/TOAT/SAMPLE-RESTART.wfn \
	--fwhm_sam 0.05 \
	--dx_wfn 0.2 \
	--orbs_tip 1 \
	--fwhm_tip 0.01 \
	--voltages -2.0 -1.8 -1.6 -1.4 -1.2 0.0 1.2 1.4 1.6 1.8 2.0 \
	--hartree_file $(SAMPLE)/TOAT/SAMPLE-HART-v_hartree-1_0.cube

# ==============================================================================
# ==============================================================================

PDOS_RELAXED= \
	--pdos_list $(TIP)/blunt/aiida-PDOS-list2-1.pdos \
				$(TIP)/blunt/aiida-PDOS-list1-1.pdos \
	--tip_shift 3.0 \
	--tip_pos_files $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPpos \
			  $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPdisp 
PDOS_UNIFORM= \
	--pdos_list $(TIP)/blunt/aiida-PDOS-list2-1.pdos \
				$(TIP)/blunt/aiida-PDOS-list1-1.pdos \
	--dx_tip 0.1 \
	--eval_region 0.0 15.0 0.0 15.0 11.5 14.0

PARA_RELAXED= \
	--pdos_list 0.15 0.5 0.0 0.5 \
	--tip_shift -3.0 \
	--tip_pos_files $(SAMPLE)/TOAT/Qo-0.12Qc0.07K0.11/PPdisp
PARA_UNIFORM= \
	--pdos_list 0.15 0.5 0.0 0.5 \
	--dx_tip 0.1 \
	--eval_region 0.0 15.0 0.0 15.0 11.5 14.0

# ==============================================================================
# ==============================================================================

TOAT_PDOS_ROT:
	$(TOAT_ARGS) $(PDOS_RELAXED) --output $(OUTPUT)/toat_tCO_pdos_rot --rotate

TOAT_PDOS_UNI:
	$(TOAT_ARGS) $(PDOS_UNIFORM) --output $(OUTPUT)/toat_tCO_pdos_uni

TOAT_PARA_ROT:
	$(TOAT_ARGS) $(PARA_RELAXED) --output $(OUTPUT)/toat_tCO_para_rot --rotate

TOAT_PARA_UNI:
	$(TOAT_ARGS) $(PARA_UNIFORM) --output $(OUTPUT)/toat_tCO_para_uni

# ==============================================================================
# ==============================================================================

clean:
	rm -rf python/__pycache__ python/*/__pycache__
