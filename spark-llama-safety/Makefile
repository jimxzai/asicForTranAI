# SPARK reference code: run the real prover. Output below comes only from GNATprove.
# Install (tested on macOS x86_64): see ../docs/INSTALL_GNAT.md
GPR := transformer.gpr

.PHONY: prove clean check-tools

check-tools:
	@command -v gnatprove >/dev/null || { echo "gnatprove not found. See ../docs/INSTALL_GNAT.md"; exit 127; }

# Exits non-zero if any check is unproved.
prove: check-tools
	gnatprove -P $(GPR) --warnings=continue
	@sed -n '/Summary of SPARK analysis/,/^Total/p' obj/gnatprove/gnatprove.out

clean:
	rm -rf obj
