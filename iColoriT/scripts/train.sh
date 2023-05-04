# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=${1:-0}
# MASTER_PORT=${2:-4885}

# # path to imagenet-1k train set and validation dataset
# DATA_PATH=''
# VAL_DATA_PATH=''
# VAL_HINT_DIR=''
# # Set the path to save checkpoints
# OUTPUT_DIR='checkpoints'
# TB_LOG_DIR='tf_logs'

# # other options
# opt=${3:-}

    
# Training epochs used for pretrained iColoriT are
# Base  - 1000 epochs
# Small - 100 epochs
# Tiny  - 25 epochs
# all with a batch size of 256.
# Other hyper-parameters follow the default numbers. 

#    

# batch_size can be adjusted according to the graphics card
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 \
    /home/yhsharaf/Desktop/iColoriT/train.py \
    --checkpoint /home/yhsharaf/Documents/iColoriT_CUB/icolorit_base_4ch_patch16_224.pth \
    --data_path /home/yhsharaf/Documents/iColoriT_Flower/102flowers_train \
    --val_data_path /home/yhsharaf/Documents/iColoriT_Flower/102flowers_validation \
    --val_hint_dir /home/yhsharaf/Documents/iColoriT_Flower/102flowers_validation_hint/1234 \
    --output_dir /home/yhsharaf/Documents/iColoriT_Flower/102flowers_500epochs_5e6 \
    --log_dir /home/yhsharaf/Documents/iColoriT_Flower/log_dir \
    --exp_name exp \
    --save_args_txt \
    # $opt
