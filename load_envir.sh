# 设置CUDA和NVSHMEM环境变量
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda
export NVSHMEM_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/nvshmem
export MPI_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/openmpi4/openmpi-4.1.5
# set mpi environment variables for OpenSHMEM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/openmpi4/openmpi-4.1.5/lib
# export NSHMEM_BOOTSTRAP_PLUGIN
# 其他环境变量设置
# ...existing code...