#!/bin/bash
#SBATCH --job-name=dhruv_clip_finetune
#SBATCH -o /scratch/users/dhruvgautam/logs/%j.out
#SBATCH -e /scratch/users/dhruvgautam/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate prsclip

cd /accounts/projects/jsteinhardt/dhruvgautam/clip_text_span/finetune

echo "Starting finetune with grayscale transform at $(date)"
HF_DATASETS_CACHE=/scratch/users/dhruvgautam/hf_datasets_cache python3 finetune.py --dataset cifar100 --batch_size 64 --epochs 30000 --model_save_interval 100 --eval_interval 10 --transform grayscale

echo "Finished grayscale run, starting invert transform at $(date)"
HF_DATASETS_CACHE=/scratch/users/dhruvgautam/hf_datasets_cache python3 finetune.py --dataset cifar100 --batch_size 64 --epochs 30000 --model_save_interval 100 --eval_interval 10 --transform invert

echo "Finished invert run, starting posterize transform at $(date)"
HF_DATASETS_CACHE=/scratch/users/dhruvgautam/hf_datasets_cache python3 finetune.py --dataset cifar100 --batch_size 64 --epochs 30000 --model_save_interval 100 --eval_interval 10 --transform posterize

echo "All runs completed at $(date)"