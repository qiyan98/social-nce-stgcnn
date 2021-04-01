# !/bin/bash
echo " Running Training EXP"


CUDA_ID=$1
if [ -z "$CUDA_ID" ]
then
  echo "CUDA_ID is empty"
  CUDA_ID=0
else
  echo "CUDA_ID is set"
fi
echo $CUDA_ID

export CUDA_VISIBLE_DEVICES=$CUDA_ID

LR=0.001
EPOCH=500
LR_RATE=10
SAMPLING=event
LOSS=nce
HORIZON=4
TAG_PREFIX=snce

# eth
WEIGHT=0.05
TEMP=0.20

python3 train.py --lr ${LR} --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag ${TAG_PREFIX}-social-stgcnn-eth --contrast_weight ${WEIGHT} --contrast_temperature ${TEMP} --contrast_horizon ${HORIZON} --contrast_sampling ${SAMPLING} --contrast_loss ${LOSS} --num_epochs ${EPOCH} --use_lrschd --lr_sh_rate ${LR_RATE} --safe_traj && echo "eth Launched." &
P0=$!

# hotel
WEIGHT=0.2
TEMP=0.20

python3 train.py --lr ${LR} --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag ${TAG_PREFIX}-social-stgcnn-hotel --contrast_weight ${WEIGHT} --contrast_temperature ${TEMP} --contrast_horizon ${HORIZON} --contrast_sampling ${SAMPLING} --contrast_loss ${LOSS} --num_epochs ${EPOCH} --use_lrschd --lr_sh_rate ${LR_RATE} --safe_traj && echo "hotel Launched." &
P1=$!

# univ
WEIGHT=0.05
TEMP=0.10

python3 train.py --lr ${LR} --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag ${TAG_PREFIX}-social-stgcnn-univ --contrast_weight ${WEIGHT} --contrast_temperature ${TEMP} --contrast_horizon ${HORIZON} --contrast_sampling ${SAMPLING} --contrast_loss ${LOSS} --num_epochs ${EPOCH} --use_lrschd --lr_sh_rate ${LR_RATE} --safe_traj && echo "univ Launched." &
P2=$!

# zara1
WEIGHT=0.2
TEMP=0.10

python3 train.py --lr ${LR} --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag ${TAG_PREFIX}-social-stgcnn-zara1 --contrast_weight ${WEIGHT} --contrast_temperature ${TEMP} --contrast_horizon ${HORIZON} --contrast_sampling ${SAMPLING} --contrast_loss ${LOSS} --num_epochs ${EPOCH} --use_lrschd --lr_sh_rate ${LR_RATE} --safe_traj && echo "zara1 Launched." &
P3=$!


# zara2
WEIGHT=0.05
TEMP=0.10

python3 train.py --lr ${LR} --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag ${TAG_PREFIX}-social-stgcnn-zara2 --contrast_weight ${WEIGHT} --contrast_temperature ${TEMP} --contrast_horizon ${HORIZON} --contrast_sampling ${SAMPLING} --contrast_loss ${LOSS} --num_epochs ${EPOCH} --use_lrschd --lr_sh_rate ${LR_RATE} --safe_traj && echo "zara2 Launched." &
P4=$!

wait $P0 $P1 $P2 $P3 $P4
