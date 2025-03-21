#!/bin/bash

# Check if build directory exists and delete it if it does
if [ -d "build" ]; then
    rm -rf build
fi


export MPIP="-f /home/j.markin/hpcproject/ff_programs/resnet_train/mpip_results/eth"
# Create build directory and navigate into it
mkdir build
cd build

# Create symbolic link for libomp.so
ln -s /opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin/libiomp5.so ./libomp.so
ln -s /home/j.markin/libjpeg-turbo-3.1.0/build/libjpeg.so.62 ./libjpeg.so.62

# Copy the model file
cp ../resnet18_cifar100.pt .
cp ../resnet152.pt .

# Copy the data directory
cp -R ../data .

# Run cmake with the specified options
cmake -DCMAKE_PREFIX_PATH=$HOME/hpcproject/pytorch-install \
      -DCMAKE_C_COMPILER=mpiicx \
      -DCMAKE_CXX_COMPILER=mpiicpx \
      -DCMAKE_EXE_LINKER_FLAGS="-qopenmp -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl" \
      -DCMAKE_CXX_FLAGS="-I${MKLROOT}/include -I ~/fastflow -I/home/j.markin/lib/cereal/include -std=c++20 -Wall -O3 -finline-functions -DNDEBUG -pthread" \
      -DTorchVision_DIR=/home/j.markin/hpcproject/vision-install/share/cmake/TorchVision/ \
      -DUSE_TORCHVISION=ON \
      -DPNG_LIBRARY=$HOME/local/lib/libpng.so \
      -DPNG_PNG_INCLUDE_DIR=$HOME/local/include \
      -DJPEG_LIBRARY=/home/j.markin/libjpeg-turbo-3.1.0/build/libjpeg.so \
      -DJPEG_INCLUDE_DIR=/home/j.markin/libjpeg-turbo-3.1.0/build ..

# Build the project
make

# Execute the load_data program
#mpirun --host localhost,localhost,localhost -n 3  gdb -ex run -ex bt --args ./train_resnet --DFF_Config=../resnet.json
I_MPI_OFI_PROVIDER=tcp I_MPI_DEBUG=5 LD_PRELOAD=/home/j.markin/mpiP_build/lib/libmpiP.so nohup mpirun -iface enp130s0f1 --host node01,node02,node03,node04,node05 -n 4 -ppn 1 ./train_resnet --DFF_Config=../resnet.json > ../eth_logs/4_workers_by2.log 2>&1 &

#I_MPI_OFI_PROVIDER=tcp I_MPI_DEBUG=5 LD_PRELOAD=/home/j.markin/mpiP_build/lib/libmpiP.so nohup mpirun -iface enp130s0f1 -n 4 -ppn 1 -f /home/j.markin/hpcproject/torchprojects/dataset_example_mpi/hostfile_eth1 ./load_data > /home/j.markin/hpcproject/torchprojects/dataset_example_mpi/logs_eth/train_model_${np}.log 2>&1 &

#mpirun --host localhost,localhost,localhost -n 2  ./train_resnet --DFF_Config=../resnet.json : --host localhost -np 1 gdb -ex run -ex bt --args ./train_resnet --DFF_Config=../resnet.json