#!/bin/bash

#SBATCH -p rise                     # partition (queue)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 1                        # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=64          # number of cores per task
#SBATCH --gres=gpu:1                # request x GPUs per node
#SBATCH --nodelist=ace              # specific nodes
#SBATCH -t 1-00:00                  # time requested (D-HH:MM)

# Print some info for context
pwd
hostname
date

echo "starting job..."

# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
source ~/.bashrc
conda activate prsclip

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN

python compute_prs.py \
    --batch_size 4 \
    --model ViT-B-16 \
    --dataset CIFAR100 \
    --device cuda \
    --pretrained laion2b_s34b_b88k \
    --data_path ~/../../../data/wong.justin/openalphaproof/ \
    --output_dir ~/../../../data/wong.justin/openalphaproof/output_dir \
    --num_workers 64