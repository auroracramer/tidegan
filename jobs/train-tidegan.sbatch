#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train-tidegan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.org
#SBATCH --output="train-tidegan-%j.out"
#SBATCH --err="train-tidegan-%j.err"

source ~/.bashrc
source activate aml

SRCDIR=''
AUDIO_A_DIR=''
AUDIO_B_DIR=''
OUTPUT_DIR=''

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/train_tidegan.py \
    --model_size 64 \
    --phase_shuffle_shift_factor 2 \
    --post_proc_filt_len 512 \
    --lrelu_alpha 0.2 \
    --valid_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --num_epochs 100 \
    --batches_per_epoch 10 \
    --ngpus 1 \
    --discriminator_updates 5 \
    --latent_dim 100 \
    --epochs_per_sample 1 \
    --sample_size 20 \
    --learning_rate 0.0001 \
    --beta_one 0.5 \
    --beta_two 0.9 \
    --regularization_factor 10.0 \
    --batchSize 64 \
    --name tidegan \
    --model_size 64 \
    --ngpus 1 \
    --latent_dim 100 \
    --lrelu-alpha 0.2 \
    --verbose \
    --phase-shuffle-shift-factor 2 \
    --batches-per-epoch 10 \
    --post_proc_filt_len 512 \
    --which_direction AtoB \
    --display_freq 400 \
    --print_freq 100 \
    --save_latest_freq 5000 \
    --save_epoch_freq 5 \
    --niter 100 \
    --niter_decay 100 \
    --beta1 0.5 \
    --beta2 0.999 \
    --lr 0.0002 \
    --lambda_A 10.0 \
    --lambda_B 10.0 \
    --lambda_identity 0.5 \
    --pool_size 50 \
    --lr_policy lambda \
    --lr_decay_iters 50 \
    $AUDIO_A_DIR \
    $AUDIO_B_DIR \
    $OUTPUT_DIR \
