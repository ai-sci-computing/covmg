# Times PETSc's fused sparse triple product MatPtAP -- the cost of FORMING a
# Galerkin coarse operator, which is the setup this paper's rediscretized
# coarsening avoids. Built against the real-scalar PETSc the project uses,
# because the complex operator is timed through its real embedding (see the
# source: E(conj z) = E(z)^T makes MatPtAP the Hermitian triple product) and
# that embedding is the configuration measured to be the faster of the two.
#
#   make            # the binary
#   make run        # export operators and time the triple product

REAL_PETSC ?= $(HOME)/.local/petsc
CXX      = /opt/homebrew/bin/mpicxx
CXXFLAGS = -O2 -std=c++17

all: galerkin_setup_bench

galerkin_setup_bench: galerkin_setup_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ \
	  -I$(REAL_PETSC)/include \
	  -L$(REAL_PETSC)/lib -lpetsc \
	  -Wl,-rpath,$(REAL_PETSC)/lib

clean:
	rm -f galerkin_setup_bench
