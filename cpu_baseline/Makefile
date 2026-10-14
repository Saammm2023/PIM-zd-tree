# CPU baselines (Pkd-tree, zd-tree) from SpaceTreeLib, which is header-only and needs g++ >= 14.
# Clone it next to this repository (tested at commit 212220f):
#   git clone --recursive https://github.com/ucrparlay/SpaceTreeLib.git ../../SpaceTreeLib
STL ?= ../../SpaceTreeLib
CXX ?= g++-14

cpu_bench: cpu_bench.cpp ../host/datagen_core.hpp
	$(CXX) -std=c++20 -O3 -march=native -mcx16 -pthread $< -o $@ \
		-I $(STL)/include -I $(STL)/include/parlaylib/include -I $(STL)/include/libmorton/include

clean:
	$(RM) cpu_bench

.PHONY: clean
