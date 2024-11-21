# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


export TRAIN_DATA_PATH=/p/vast1/MLdata/project_gutenberg/processed
export VALID_DATA_PATH=/p/vast1/MLdata/project_gutenberg/processed/val
#export NCCL_SOCKET_IFNAME=hsi0
firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
#firsthost=$(hostname)
export MASTER_ADDRESS=$firsthost
#export MAIN_OPRT=25855
devices=4
nnodes=2
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo $MASTER_ADDRESS

MODEL_NAME='tiny_LLaMA_135M_2k'
srun torchrun \
    --nnodes $nnodes \
    --rdzv-id $RANDOM \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDRESS \
    --nproc-per-node $devices \
    pretrain/myllama.py --devices_ $devices --train_data_dir $TRAIN_DATA_PATH  --val_data_dir $VALID_DATA_PATH --model_name my_LLaMA_7b\
    #pretrain/tinyllama.py --devices_ $devices --train_data_dir $TRAIN_DATA_PATH  --val_data_dir $VALID_DATA_PATH --model_name $MODEL_NAME\

#--standalone \
