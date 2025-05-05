# ===== IMPORTS =====
# Standard libraries
import json
import argparse

# Third-party libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import wandb
import open_clip
from datasets import load_dataset, Dataset

# Local modules
from data import process_birds, process_imagenet, process_cifar100, process_cifarc
from modules import ScaledMultiheadAttention, wrap_multihead_attention

# ===== ARGUMENT PARSING =====
##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='zero-shot', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K', help='CLIP model name')
parser.add_argument('--transform', type=str, default=None, help='Grayscale images')
parser.add_argument('--semantic_shift', type=str, default='', help='Semantic shift to apply')
parser.add_argument('--semantic_shuffle', type=bool, default=False, help='Shuffle classes')
parser.add_argument('--wrap', type=bool, default=False, help='Semantic shift to apply')
parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint file')
parser.add_argument('--checkpoint_epoch', type=int, default=0, help='Epoch of the checkpoint to load')
parser.add_argument('--shift_shuffle', type=int, default=0, help='Number of classes to shift and shuffle')

args = parser.parse_args()

# ===== MODEL CONFIGURATION =====
# Set device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP model and preprocessing transforms
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
tokenizer = open_clip.get_tokenizer(args.clip_model)

# Optionally wrap the model's multihead attention layers for semantic shift
if args.wrap:
    model = wrap_multihead_attention(model)

# Load checkpoint if specified
if args.checkpoint:
    run = wandb.init()
    artifact = wandb.use_artifact(args.checkpoint, type='model')
    artifact_dir = artifact.download()
    checkpoint = torch.load(f"{artifact_dir}/checkpoint_epoch_{args.checkpoint_epoch}.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = torch.optim.Adam(model.parameters())
model.to(device)

loss_fn = nn.CrossEntropyLoss()

# ===== IMAGE PREPROCESSING =====
# Load and process dataset based on user input
if args.dataset == 'birdsnap':
    ds = Dataset.from_file("../birdsnap_dataset/train/data-00001-of-00139.arrow")
    json_contents = json.load(open("./birdsnap_prompts.json"))
    ds, classes_to_index, index_to_classes, captions = process_birds(ds, json_contents)
elif args.dataset == 'imagenet_sketch':
    ds = load_dataset("imagenet_sketch", split="train", trust_remote_code=True)
    ds = ds.select(range(1000))
    json_contents = json.load(open("./imagenet_prompts.json"))
    ds, classes_to_index, index_to_classes, captions = process_imagenet(ds, json_contents, transform=args.transform)
elif args.dataset == 'cifar100':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = load_dataset("cifar100", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42)
    json_contents = json.load(open("./cifar100_prompts.json"))
    ds, classes_to_index, index_to_classes, captions = process_cifar100(
        ds, json_contents, preprocess_val,
        transform=args.transform,
        semantic_shift=args.semantic_shift,
        semantic_shuffle=args.semantic_shuffle,
        shift_shuffle=args.shift_shuffle
    )
# elif args.dataset == 'cifarc':
#     ds = load_dataset("randall-lab/cifar100-c", split="test", trust_remote_code=True)
#     ds = ds.shuffle(seed=42)
#     ds = ds.select(range(1000))
#     json_contents = json.load(open("./cifar100_prompts.json"))
#     ds, classes_to_index, index_to_classes, captions = process_cifarc(ds, json_contents, preprocess_val, transform=args.transform, semantic_shift=args.semantic_shift, semantic_shuffle=args.semantic_shuffle, shift_shuffle=args.shift_shuffle)
#     # "randall-lab/cifar100-c", split="test", trust_remote_code=True

# Split dataset into train and test sets
dataset_dict = ds.train_test_split(test_size=0.2, seed=42)
train_ds = dataset_dict['train']
test_ds = dataset_dict['test']

# ===== WANDB CONFIGURATION =====
wandb.init(project=args.wandb_project, config=args)

# Log sample images with their labels to WandB for visualization
images = [wandb.Image(ds[i]["image"], caption=f"Label: {index_to_classes[ds[i]['index_label']]}") for i in range(5)]
wandb.log({"images": images})

# ===== TRAINING LOOP WITH LEAST SQUARES REGRESSION =====

from torch.nn.functional import cosine_similarity

train_dataloader = DataLoader(train_ds, batch_size=100, shuffle=True)

total_loss = 0.0
total_samples = 0
pbar = tqdm(train_dataloader, desc="Least Squares Projection")
pbar.set_description(f"Running Least Squares Projection on {args.dataset}")

for sample in pbar:
    with torch.no_grad(), autocast(device_type=device.type):
        image = sample["image"].to(device, dtype=torch.float16, non_blocking=True)
        transformed_image = transforms.Grayscale(num_output_channels=3)(image).to(device, dtype=torch.float16, non_blocking=True)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        transformed_image_features = model.encode_image(transformed_image)
        transformed_image_features /= transformed_image_features.norm(dim=-1, keepdim=True)

        ones = torch.ones(transformed_image_features.size(0), 1, dtype=transformed_image_features.dtype, device=device)
        X_aug = torch.cat([transformed_image_features, ones], dim=1)

        X_aug = X_aug.to(torch.float32)
        image_features = image_features.to(torch.float32)

        W = torch.linalg.lstsq(X_aug, image_features).solution
        A = W[:-1, :]
        b = W[-1, :]

        pred_features = transformed_image_features @ A + b
        loss = 1 - cosine_similarity(pred_features, image_features, dim=-1).mean()

        total_loss += loss.item() * image_features.size(0)
        total_samples += image_features.size(0)
        avg_loss = total_loss / total_samples

        pbar.set_postfix({"Cosine Similarity Loss": avg_loss})
        wandb.log({"avg_cosine_similarity_loss": avg_loss})

print(f"Final Average Cosine Similarity Loss: {avg_loss:.6f}")

# ===== TEST LOOP =====
test_dataloader = DataLoader(test_ds, batch_size=100)

test_total_loss = 0.0
test_total_samples = 0
test_pbar = tqdm(test_dataloader, desc="Evaluating on Test Set")

for sample in test_pbar:
    with torch.no_grad(), autocast(device_type=device.type):
        image = sample["image"].to(device, dtype=torch.float16)
        transformed_image = transforms.Grayscale(num_output_channels=3)(image).to(device, dtype=torch.float16)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        transformed_image_features = model.encode_image(transformed_image)
        transformed_image_features /= transformed_image_features.norm(dim=-1, keepdim=True)

        pred_features = transformed_image_features @ A + b
        loss = 1 - cosine_similarity(pred_features, image_features, dim=-1).mean()

        test_total_loss += loss.item() * image_features.size(0)
        test_total_samples += image_features.size(0)
        test_avg_loss = test_total_loss / test_total_samples

        test_pbar.set_postfix({"Test Cosine Similarity Loss": test_avg_loss})

wandb.log({"test_avg_cosine_similarity_loss": test_avg_loss})
print(f"Final Test Cosine Similarity Loss: {test_avg_loss:.6f}")

# ===== EXAMPLE RUN =====
# Example command to run this script:
# python least-squares.py --dataset cifar100 --clip_model hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K --transform grayscale --wandb_project test_projection