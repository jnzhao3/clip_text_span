#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=64 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:8
# note: nodelist says you want ALL of the nodes, not ANY of the nodes
# use --exclude instead if you want to limit to a subset of machines
#SBATCH --nodelist=ace # if you need specific nodes
##SBATCH --exclude=blaze,freddie # nodes not yet on SLURM-only
#SBATCH -t 1-00:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
##SBATCH -D /home/eecs/drothchild/slurm
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
##SBATCH -o slurm.%N.%j.out # STDOUT
##SBATCH -e slurm.%N.%j.err # STDERR
# if you want to get emails as your jobs run/fail
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=shuoyuan@berkeley.edu # Where to send mail 

# print some info for context
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
    --model ViT-L-14 \
    --dataset CIFAR100 \
    --device cuda \
    --pretrained laion2b_s32b_b82k \
    --data_path ~/../../../data/wong.justin/openalphaproof/ \
    --output_dir ~/../../../data/wong.justin/openalphaproof/output_dir \
    --num_workers 128