#!/bin/bash

#SBATCH -p rise                     # partition (queue)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 1                        # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=64          # number of cores per task
#SBATCH --gres=gpu:1                # request 8 GPUs per node
#SBATCH --nodelist=ace              # specific nodes
#SBATCH -t 1-00:00                  # time requested (D-HH:MM)

# Print some info for context
pwd
hostname
date

echo "starting job..."

# Activate your virtual environment
source ~/.bashrc
conda activate prsclip

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Run the Python script
python compute_complete_text_set.py \
    --model ViT-B-16 \
    --dataset CIFAR100 \
    --device cuda \
    --input_dir ~/../../../data/wong.justin/openalphaproof/output_dir \
    --output_dir ~/../../../data/wong.justin/openalphaproof/output_dir \
    --text_descriptions image_descriptions_general \
    --num_workers 64 \
    --num_of_last_layers 4
