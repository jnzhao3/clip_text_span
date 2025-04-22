"""
script to compute the saliency maps of images. 
"""

## Imports
import numpy as np
import torch
from PIL import Image, ImageOps
import os.path
import argparse
from pathlib import Path
import cv2
import heapq
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import einops
from torchvision import datasets, transforms
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.visualization import image_grid, visualization_preprocess
from prs_hook import hook_prs_logger
from matplotlib import pyplot as plt
import json
import matplotlib.cm as cm
from collections import defaultdict
import wandb
from tqdm import tqdm

# ==== Argument Parsing ====
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset to use')
parser.add_argument('--data_dir', type=str, default='./datasets', help='Path to dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
parser.add_argument('--images_per_class', type=int, default=1, help='Number of images per fine class to visualize')
parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
args = parser.parse_args()

wandb.init(project="CIFAR100_Saliency", name="saliency_map_run", config=vars(args))

## Hyperparameters
device = args.device
pretrained = 'laion2b_s34b_b88k' 

# imagenet_path = '/data/cifar-100-python/' # only needed for the nn search

# ==== Model Setup ====
if args.model_path and args.model_path.endswith((".pt", ".pth")):
    model, _, preprocess = create_model_and_transforms("ViT-B-16", pretrained=pretrained)
    checkpoint = torch.load(args.model_path, map_location=args.device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
else:
    model, _, preprocess = create_model_and_transforms(args.model_path, pretrained=pretrained)

model.to(device)
model.eval()
context_length = model.context_length
vocab_size = model.vocab_size
tokenizer = get_tokenizer("ViT-B-16")

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Context length:", context_length)
print("Vocab size:", vocab_size)
print("Len of res:", len(model.visual.transformer.resblocks))

prs = hook_prs_logger(model, device)

# ==== Data Loading ====
## Load image
if args.dataset == 'CIFAR100':
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])
    dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    coarse_targets = dataset.targets
    fine_targets = dataset.targets
    coarse_label_map = dataset.classes  # CIFAR-100 doesn't expose coarse labels natively
    label_to_indices = defaultdict(list)
    for idx, (_, target) in enumerate(dataset):
        label_to_indices[target].append(idx)

    images = []
    image_pils = []
    selected_labels = []

    for label, indices in sorted(label_to_indices.items())[:10]:
        for idx in indices[:args.images_per_class]:
            images.append(dataset[idx][0])
            image_pils.append(dataset.data[idx])
            selected_labels.append(label)  # record the correct fine label

    images = torch.stack(images)

model_id = args.model_path.replace("/", "_").replace(".", "_") if args.model_path else "ViT-B-16"
output_dir = Path(f"CIFAR_100_saliency_maps/{model_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# ==== Inference and Visualization ====
for i in tqdm(range(len(images)), desc="Processing images"):
    image = images[i].unsqueeze(0).to(device)
    image_pil = Image.fromarray(image_pils[i])
    _ = plt.imshow(image_pil)

    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image, attn_method='head', normalize=False)
        attentions, mlps = prs.finalize(representation)

    lines = [f"An image of a {dataset.classes[selected_labels[i]]}"]
    texts = tokenizer(lines).to(device)  # tokenize
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)


    attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
    attention_map = F.interpolate(
        einops.rearrange(attention_map, '(B N M) C -> B C N M', N=14, M=14, B=1),
        scale_factor=model.visual.patch_size[0],
        mode='bilinear'
    ).to(device)
    attention_map = attention_map[0].detach().cpu().numpy()

    avg_saliency = attention_map.mean(axis=0)
    max_saliency = attention_map.max(axis=0)

    for agg_type, agg_map in [("avg", avg_saliency), ("max", max_saliency)]:
        v = agg_map - np.mean(agg_map)
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-5)
        heatmap = np.uint8(cm.jet(v_norm) * 255)
        label_dir = output_dir / dataset.classes[selected_labels[i]]
        label_dir.mkdir(parents=True, exist_ok=True)
        type_dir = label_dir / agg_type
        type_dir.mkdir(parents=True, exist_ok=True)

        heatmap_img = Image.fromarray(heatmap).convert("RGBA").resize((224, 224))
        orig_img_resized = image_pil.resize((224, 224)).convert("RGB")

        combined = Image.new("RGB", (448, 224))
        combined.paste(orig_img_resized, (0, 0))
        heatmap_rgb = heatmap_img.convert("RGB")
        heatmap_autocontrast = ImageOps.autocontrast(heatmap_rgb)
        combined.paste(heatmap_autocontrast, (224, 0))

        combined_path = type_dir / f"{dataset.classes[selected_labels[i]]}_idx{i}_saliency_comparison.png"
        orig_img_path = type_dir / f"{dataset.classes[selected_labels[i]]}_idx{i}_original.png"
        heatmap_img_path = type_dir / f"{dataset.classes[selected_labels[i]]}_idx{i}_heatmap.png"

        combined.save(combined_path)
        orig_img_resized.save(orig_img_path)
        heatmap_autocontrast.save(heatmap_img_path)

        wandb.log({
            f"{dataset.classes[selected_labels[i]]}_{agg_type}_comparison_{i}": wandb.Image(combined, caption=f"{dataset.classes[selected_labels[i]]} [{i}] - ({agg_type} comparison)"),
            f"{dataset.classes[selected_labels[i]]}_{agg_type}_original_{i}": wandb.Image(orig_img_resized, caption=f"{dataset.classes[selected_labels[i]]} [{i}] - original"),
            f"{dataset.classes[selected_labels[i]]}_{agg_type}_heatmap_{i}": wandb.Image(heatmap_autocontrast, caption=f"{dataset.classes[selected_labels[i]]} [{i}] - heatmap")
        })

wandb.finish()
