#!/bin/bash
rocm_version=6.2.1
source /etc/profile.d/z00_lmod.sh
module swap PrgEnv-cray PrgEnv-gnu
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load cray-pmi/6.1.15
module load cray-libpals/1.2.11
module load libfabric/2.1

# Clear existing env vars
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# Essential NCCL settings
export NCCL_DEBUG=WARN  # Changed from INFO to WARN to reduce output
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsi0
export NCCL_NET_GDR_LEVEL=5

# AMD specific settings
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HSA_FORCE_FINE_GRAIN_PCIE=0

# Uncomment these 4 lines to go fast!
plugin_loc=/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.2.1/install/lib
export LD_LIBRARY_PATH=${plugin_loc}:$LD_LIBRARY_PATH  # point to directory with plugin library
export NCCL_NET_GDR_LEVEL=3  # enable GPU Direct RDMA communication
export FI_CXI_ATS=0 # enable GPU Direct RDMA communication


# original launch script starts here
export TRAIN_DATA_PATH=/p/vast1/MLdata/project_gutenberg/processed
export VALID_DATA_PATH=/p/vast1/MLdata/project_gutenberg/processed/val
#export NCCL_SOCKET_IFNAME=hsi0
firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
#firsthost=$(hostname)
export MASTER_ADDRESS=$firsthost
#export MAIN_OPRT=25855
devices=4
nnodes=64
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo $MASTER_ADDRESS

#MODEL_NAME='tiny_LLaMA_135M_2k'
srun torchrun \
    --nnodes $nnodes \
    --rdzv-id $RANDOM \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDRESS \
    --nproc-per-node $devices \
    pretrain/myllama.py --devices_ $devices --train_data_dir $TRAIN_DATA_PATH  --val_data_dir $VALID_DATA_PATH --model_name my_LLaMA_7b\
    #pretrain/tinyllama.py --devices_ $devices --train_data_dir $TRAIN_DATA_PATH  --val_data_dir $VALID_DATA_PATH --model_name $MODEL_NAME\

#--standalone \
