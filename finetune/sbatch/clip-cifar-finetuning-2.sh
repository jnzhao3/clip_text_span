#!/bin/bash
#SBATCH --job-name=jenn
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --array=1-1%16

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=3
JOB_N=3

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
module load gnu-parallel
source ~/.bashrc
micromamba activate prsclip

declare -a commands=(
 [1]='WANDB_DIR=/global/scratch/users/jenniferzhao/wandb_logs HF_DATASETS_CACHE=/global/scratch/users/jenniferzhao/hf_datasets_cache python3 finetune.py --dataset cifar100 --batch_size 32 --epochs 30000 --model_save_interval 2 --eval_interval 2 --transform grayscale'
 [2]='WANDB_DIR=/global/scratch/users/jenniferzhao/wandb_logs HF_DATASETS_CACHE=/global/scratch/users/jenniferzhao/hf_datasets_cache python3 finetune.py --dataset cifar100 --batch_size 32 --epochs 30000 --model_save_interval 2 --eval_interval 2 --transform invert'
 [3]='WANDB_DIR=/global/scratch/users/jenniferzhao/wandb_logs HF_DATASETS_CACHE=/global/scratch/users/jenniferzhao/hf_datasets_cache python3 finetune.py --dataset cifar100 --batch_size 32 --epochs 30000 --model_save_interval 2 --eval_interval 2 --transform posterize'
)

cd /global/home/users/jenniferzhao/clip_text_span/clip_text_span/finetune

parallel --delay 20 --linebuffer -j 3 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
