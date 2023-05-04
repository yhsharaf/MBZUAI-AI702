# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=${1:-0}

# # path to model and validation dataset
# MODEL_PATH='/home/yhsharaf/Documents/iColorTest/icolorit_base_4ch_patch16_224.pth'
# VAL_DATA_PATH='/home/yhsharaf/Documents/iColorTest/200cub'
# VAL_HINT_DIR='/home/yhsharaf/Documents/iColorTest/200cub_hint/1234'
# # Set the path to save checkpoints
# PRED_DIR='/home/yhsharaf/Documents/iColorTest/200cub_pred'

# # other options
# opt=${2:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/infer.py\
    --model_path=/home/yhsharaf/Documents/iColoriT_Flower/102flowers_500epochs_5e6/icolorit_base_4ch_patch16_224/exp_230427_010224/checkpoint-99.pth \
    --val_data_path=/home/yhsharaf/Documents/iColoriT_Flower/102flowers_test \
    --val_hint_dir=/home/yhsharaf/Documents/iColoriT_Flower/102flowers_test_hint/1234 \
    --pred_dir=/home/yhsharaf/Documents/iColoriT_Flower/102flowers_500epochs_5e6/icolorit_base_4ch_patch16_224/exp_230427_010224/predicted \