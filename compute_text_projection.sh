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

# Activate your virtual environment
source ~/.bashrc
conda activate prsclip

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Hugging Face authentication
export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN

# Run the Python script
python compute_text_projection.py \
    --model ViT-B-16 \
    --dataset CIFAR100 \
    --pretrained laion2b_s34b_b88k \
    --device cuda \
    --output_dir ~/../../../data/wong.justin/openalphaproof/output_dir
