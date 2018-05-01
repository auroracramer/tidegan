#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train-tidegan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output="train-tidegan-%j.out"
#SBATCH --err="train-tidegan-%j.err"

source ~/.bashrc
source activate aml

SRCDIR="/home/jtc440/dev/tidegan"
AUDIO_A_DIR="/beegfs/jtc440/aml/nsynth/nsynth-train/brass_acoustic"
AUDIO_B_DIR="/beegfs/jtc440/aml/nsynth/nsynth-train/reed_acoustic"
OUTPUT_DIR="/scratch/jtc440/cyclegan/brass-reed-nsynth"

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/train_tidegan.py \
    --batchSize 64 \
    --name tidegan \
    --model_size 64 \
    --ngpus 1 \
    --latent_dim 100 \
    --lrelu_alpha 0.2 \
    --verbose \
    --phase_shuffle_shift_factor 2 \
    --batches_per_epoch 10 \
    --post_proc_filt_len 512 \
    --which_direction AtoB \
    --display_freq 400 \
    --print_freq 100 \
    --save_latest_freq 5000 \
    --save_epoch_freq 5 \
    --niter 500 \
    --niter_decay 500 \
    --beta1 0.5 \
    --beta2 0.9 \
    --lr 0.0001 \
    --gan-loss wgan-wp \
    --lambda_A 10.0 \
    --lambda_B 10.0 \
    --lambda_identity 0.5 \
    --pool_size 50 \
    --lr_policy lambda \
    --lr_decay_iters 50 \
    "$AUDIO_A_DIR" \
    "$AUDIO_B_DIR" \
    "$OUTPUT_DIR"
