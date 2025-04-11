# CS 182 Final Project

## Download the Birds

```
conda activate prsclip
pip install open-clip-torch
pip install datasets
```

## Download the Birds
```
from datasets import load_dataset

ds = load_dataset("sasha/birdsnap")
```

## Access Imagenet

* Request access at https://huggingface.co/datasets/imagenet-1k.
* Create access token from huggingface with access to imagenet-1k
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