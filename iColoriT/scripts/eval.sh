# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=${1:-0}

PRED_DIR='/home/yhsharaf/Documents/iColoriT_Flower/102flowers_500epochs_5e6/icolorit_base_4ch_patch16_224/exp_230427_010224/predicted'
GT_DIR='/home/yhsharaf/Documents/iColoriT_Flower/102flowers_test/imgs'
NUM_HINT=${2:-0}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-1}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-2}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-5}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-10}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-20}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt


NUM_HINT=${2:-50}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-100}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt

NUM_HINT=${2:-200}

# other options
# opt=${3:-}

# batch_size can be adjusted according to the graphics card
python /home/yhsharaf/Desktop/iColoriT/evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    # $opt


