# CS 182 Final Project

## Download the Birds

```
conda activate prsclip
pip install open-clip-torch
pip install datasets
pip install wandb
```

## Download the Birds
```
from datasets import load_dataset

ds = load_dataset("sasha/birdsnap")
```

## Access Imagenet

* Request access at https://huggingface.co/datasets/imagenet-1k.
* Create access token from huggingface with access to imagenet-1k.
```
huggingface-cli login
```
* Use access token from huggingface to authenticate.
* Run `python zero-shot.py --dataset imagenet`. Takes a while the first time.

```
python zero-shot.py
```

## Data Processing Functions
Format the data to be like:

```
image = sample["image"]
label = sample["label"]
index_label = sample["index_label"]
```

* These are the only required columns for now.

## WandB

* Be sure to paste in your authentication code.
* Should log the first five images of the dataset.

## The Finetuning Script
* These are the arguments to the Finetuner class:

```
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--model_save_interval', type=int, default=5, help='Model save interval')
parser.add_argument('--eval_interval', type=int, default=2, help='Training evaluation interval')
```

* So an example run command:
```
python finetune.py --dataset imagenet --batch_size 32 --epochs 30 --model_save_interval 10 --eval_interval 10 --run_name name-of-wandb-run
```

## Transforms

* Can now add different transformations to finetune or zero-shot, use:

```
--transform grayscale
--transform invert
--transform posterize
```

* Note: currently, can only use one transformation at a time.