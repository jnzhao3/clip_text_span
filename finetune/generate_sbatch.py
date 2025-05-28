import os
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-j", default=4, type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--limit", default=16, type=int)

args, unknown = parser.parse_known_args()

name = args.name
limit = args.limit

print(unknown)


def parse(args):
    prefix = ""
    for index in range(len(args)):
        prefix += " "
        arg = args[index]
        i = arg.find("=")
        if i == -1:
            content = arg
        else:
            prefix += arg[: i + 1]
            content = arg[i + 1 :]

        if "," in content:
            elements = content.split(",")
            for r in parse(args[index + 1 :]):
                for element in elements:
                    yield prefix + element + r
            return
        else:
            prefix += content
    yield prefix


python_command_list = list(map(lambda x: x.replace("|", ","), parse(unknown)))

num_jobs = len(python_command_list)

num_arr = (num_jobs - 1) // args.j + 1

print("\n".join(python_command_list))

path = os.getcwd()

d_str = "\n ".join(
    [
        "[{}]='{}'".format(i + 1, command[1:])
        for i, command in enumerate(python_command_list)
    ]
)

sbatch_str = f"""#!/bin/bash
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
#SBATCH --array=1-{num_arr}%{limit}

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={args.j}
JOB_N={num_jobs}

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
module load gnu-parallel
source ~/.bashrc
micromamba activate prsclip

declare -a commands=(
 {d_str}
)

cd {path}

parallel --delay 20 --linebuffer -j {args.j} {{1}} ::: \"${{commands[@]:$COM_ID_S:$PARALLEL_N}}\"
"""

with open(f"sbatch/{name}.sh", "w") as f:
    f.write(sbatch_str)