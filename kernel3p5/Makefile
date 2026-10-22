# Reference 3.5-bit kernel: tests and CPU benchmark. Needs gfortran only.
ifeq ($(origin FC),default)
FC := gfortran
endif
BUILD   := build
TESTFLG := -O2 -Wall -Wextra -Wno-integer-division -Wno-compare-reals -fcheck=all -fbacktrace
BENCHFL := -O3 -march=native

.PHONY: all test bench clean
all: test

$(BUILD):
	mkdir -p $(BUILD)

test: | $(BUILD)
	$(FC) $(TESTFLG) -J$(BUILD) -o $(BUILD)/test_pack3p5 pack3p5.f90 test_pack3p5.f90
	./$(BUILD)/test_pack3p5
	$(FC) $(TESTFLG) -J$(BUILD) -o $(BUILD)/test_mxfp4 mxfp4.f90 test_mxfp4.f90
	./$(BUILD)/test_mxfp4

bench: | $(BUILD)
	$(FC) $(BENCHFL) -J$(BUILD) -o $(BUILD)/bench_matmul pack3p5.f90 mxfp4.f90 bench_matmul.f90
	./bench_matmul_env.sh ./$(BUILD)/bench_matmul

clean:
	rm -rf $(BUILD)
