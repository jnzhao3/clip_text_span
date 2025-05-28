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
#SBATCH --array=1-2%16

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=2
JOB_N=3

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
module load gnu-parallel
source ~/.bashrc
micromamba activate prsclip

declare -a commands=(
 [1]='python3 finetune/finetune.py --dataset cifar100 --batch_size 64 --epochs 30000 --model_save_interval 100 --eval_interval 10 --transform grayscale'
 [2]='python3 finetune/finetune.py --dataset cifar100 --batch_size 64 --epochs 30000 --model_save_interval 100 --eval_interval 10 --transform invert'
 [3]='python3 finetune/finetune.py --dataset cifar100 --batch_size 64 --epochs 30000 --model_save_interval 100 --eval_interval 10 --transform posterize'
)

cd /global/home/users/jenniferzhao/clip_text_span/clip_text_span

parallel --delay 20 --linebuffer -j 2 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
