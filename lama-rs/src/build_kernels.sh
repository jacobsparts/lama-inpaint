#!/bin/sh
# Rebuild the embedded GPU kernel image.
#
# The image is a fatbin carrying real SASS for every listed compute capability,
# so the binary runs on those architectures without the driver having to JIT
# anything.  PTX is deliberately NOT included: nvcc 12.4 emits `.version 8.4`
# PTX for every target, and older drivers refuse to JIT PTX newer than they
# understand (the 535 driver caps out at ISA 8.2 and fails with error 98).
# Adding an architecture therefore means adding a -gencode line here and the
# matching entry in KERNEL_ARCHS in cuda.rs.
#
#   sh src/build_kernels.sh
#
set -e
cd "$(dirname "$0")/.."
nvcc   -gencode arch=compute_61,code=sm_61   -gencode arch=compute_75,code=sm_75   -gencode arch=compute_80,code=sm_80   -gencode arch=compute_86,code=sm_86   -gencode arch=compute_89,code=sm_89   -gencode arch=compute_90,code=sm_90   -fatbin src/kernels.cu -o src/kernels.fatbin
cuobjdump src/kernels.fatbin | grep -c 'arch = ' | xargs echo "architectures in image:"
