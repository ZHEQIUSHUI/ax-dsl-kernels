# Board programs (aarch64), cross-compiled against the AX8860 BSP.
#   make BSP=<bsp>/arm64_glibc CROSS=<arm-gnu-toolchain-13.2>/bin/aarch64-none-linux-gnu-
# The BSP libraries need glibc 2.38, so the Arm GNU toolchain 13.2.Rel1 (or newer) is required.
BSP   ?= $(error set BSP to the AX8860 BSP directory, e.g. .../arm64_glibc)
CROSS ?= aarch64-none-linux-gnu-
CXX    = $(CROSS)g++
FLAGS  = -O2 -Wall -I$(BSP)/include
LIBS   = -L$(BSP)/lib -lax_engine -lax_interpreter -lax_sys -lpthread -ldl -Wl,--allow-shlib-undefined
OUT   ?= build

PROGS = $(OUT)/axrun $(OUT)/iojson $(OUT)/par_test $(OUT)/run_alias $(OUT)/kv_inplace $(OUT)/inplace_test \
        $(OUT)/mc_stress $(OUT)/mc_alt

all: $(PROGS)

$(OUT)/axrun:        tools/board/axrun.cc
$(OUT)/iojson:       tools/board/iojson.c
$(OUT)/par_test:     tools/board/par_test.cc
$(OUT)/run_alias:    tools/board/run_alias.cc
$(OUT)/kv_inplace:   tools/board/kv_inplace.cc
$(OUT)/inplace_test: tools/board/inplace_test.cc
$(OUT)/mc_stress:    tools/board/mc_stress.cc
$(OUT)/mc_alt:       tools/board/mc_alt.cc

$(OUT)/par_test: STD = -std=c++20
$(PROGS): STD ?= -std=c++17

$(OUT)/iojson: tools/board/iojson.c
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -x c++ -std=c++17 $< $(LIBS) -o $@

$(filter-out $(OUT)/iojson,$(PROGS)):
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) $(STD) $< $(LIBS) -o $@
