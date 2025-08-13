# 设置CUDA和NVSHMEM环境变量
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda
export NVSHMEM_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/nvshmem
export MPI_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/hpcx/hpcx-2.20/ompi
# set mpi environment variables for OpenSHMEM
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
# export NSHMEM_BOOTSTRAP_PLUGIN
# 其他环境变量设置
# ...existing code...
