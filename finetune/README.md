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