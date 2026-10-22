# SPARK-LLaMA-Safety: run the real prover. Output below comes only from the prover.
GPR := transformer.gpr

.PHONY: prove build clean check-tools

check-tools:
	@command -v gnatprove >/dev/null || { echo "gnatprove not found. See ../docs/INSTALL_GNAT.md"; exit 127; }

# Exits non-zero if any check is unproven. Read the summary in gnatprove/gnatprove.out.
prove: check-tools
	gnatprove -P $(GPR) --level=2 --report=all --warnings=error
	@echo "--- summary ---"
	@sed -n '/Summary of SPARK analysis/,$$p' gnatprove/gnatprove.out

build: check-tools
	gprbuild -P $(GPR)

clean:
	rm -rf obj bin gnatprove *.o *.ali
