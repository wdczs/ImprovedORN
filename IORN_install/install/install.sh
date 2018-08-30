#!/bin/bash
HOME=$(pwd)
export PATH=/usr/local/cuda/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "Compiling cuda kernels..."
cd ./iorn/src
# rm liborn_kernel.cu.o
nvcc -c -o liborn_kernel.cu.o liborn_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
echo "Installing extension..."
cd ..
cd ..
python setup.py clean && python setup.py install
