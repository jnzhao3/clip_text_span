# CS 182 Final Project

## 🔧 Environment Setup

Activate the conda environment and install dependencies:

```bash
conda activate prsclip
pip install open-clip-torch datasets wandb
```

---

## 📁 Dataset Setup

### Birdsnap

Download the dataset using Hugging Face Datasets:

```python
from datasets import load_dataset

ds = load_dataset("sasha/birdsnap")
```

### ImageNet-Sketch

To access and use ImageNet-Sketch:

1. **Request Access**: [https://huggingface.co/datasets/imagenet-1k](https://huggingface.co/datasets/imagenet-1k)
2. **Authenticate**:
   ```bash
   huggingface-cli login
   ```
3. **Run Zero-Shot Script**:
   ```bash
   python zero-shot.py --dataset imagenet_sketch
   ```

> Note: This may take a while to run the first time.

---

## 📊 Data Format Requirements

Ensure that each sample contains the following keys:

```python
image = sample["image"]
label = sample["label"]
index_label = sample["index_label"]
```

These are the **required columns**.

---

## 🧪 Logging with Weights & Biases

- Paste your **WandB authentication code** when prompted.
- The script will log the **first five images** of the dataset automatically.

---

## 🏋️ Finetuning Script

### Script Arguments

The `finetune.py` script accepts the following arguments:

```python
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--model_save_interval', type=int, default=5, help='Model save interval')
parser.add_argument('--eval_interval', type=int, default=2, help='Training evaluation interval')
```

### Example Run

```bash
python finetune.py --dataset imagenet --batch_size 32 --epochs 30 --model_save_interval 10 --eval_interval 10 --run_name name-of-wandb-run
```

---

## 🎨 Supported Image Transformations

Add a visual transformation using the `--transform` argument:

```bash
--transform grayscale
--transform invert
--transform posterize
```

> Note: Only one transformation can be applied at a time.