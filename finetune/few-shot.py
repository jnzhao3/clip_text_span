"""
Few-Shot Image Classification with CLIP

This script computes few-shot classification prompts for each class in a dataset
(e.g., CIFAR-100) using image embeddings from a CLIP model. It constructs prompts 
using the 3 most visually similar images (based on cosine similarity) and evaluates 
model accuracy using these prompts. Results and metrics are logged to Weights & Biases (W&B).
"""

import torch
from PIL import Image
import open_clip
from datasets import Dataset, Image, load_dataset
import datasets
import json
import argparse
import wandb
from tqdm import tqdm
from data import process_birds, process_imagenet, process_cifar100
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import torchvision.transforms as transforms

##===== END OF IMPORTS =====##

# Utility functions are now moved below the import section for better readability and modular structure
# ==== UTILITY FUNCTIONS ====

def compute_dendrogram(embeddings, labels):
    """Generate and log a dendrogram of label embeddings using hierarchical clustering."""
    linkage_matrix = sch.linkage(embeddings, method='ward')
    plt.figure(figsize=(12, 6))
    dendro = sch.dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title("Dendrogram of Label Embeddings")
    plt.xlabel("Class Label")
    plt.ylabel("Distance")
    wandb.log({"Dendrogram": wandb.Image(plt)})
    plt.close()

def perform_clustering(embeddings, n_clusters=5):
    """Perform agglomerative clustering on embeddings and return cluster IDs."""
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return clustering.fit_predict(embeddings)

def compute_tsne(embeddings, labels, groups):
    """Compute and log a 2D t-SNE projection of the label embeddings with group annotations."""
    tsne_perplexity = min(5, len(labels) - 1)
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init='random', random_state=42)
    coords = tsne.fit_transform(embeddings)
    tsne_table = wandb.Table(columns=["x", "y", "label", "group"])
    for i, (x, y) in enumerate(coords):
        tsne_table.add_data(x, y, labels[i], groups[i])
    wandb.log({
        "class_embedding_tsne": tsne_table,
        "t-SNE Hover Plot": wandb.plot_table("wandb/point", tsne_table, {"x": "x", "y": "y"}, {"class label": "label", "group": "group"})
    })
    return coords

def compute_pca(embeddings, labels, groups):
    """Compute and log a 2D PCA projection of the label embeddings with group annotations."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    embedding_table = wandb.Table(columns=["x", "y", "label", "group"])
    for i, (x, y) in enumerate(coords):
        embedding_table.add_data(x, y, labels[i], groups[i])
    wandb.log({
        "class_embedding_projection": embedding_table,
        "PCA Hover Plot": wandb.plot_table("wandb/point", embedding_table, {"x": "x", "y": "y"}, {"class label": "label", "group": "group"})
    })
    return coords

def compute_per_class_accuracy(ds, model, preprocess_fn, text_features, class_labels):
    """Calculate per-class accuracy over the dataset."""
    from collections import defaultdict
    class_counts = defaultdict(int)
    class_correct = defaultdict(int)

    for sample in tqdm(ds, desc="Computing per-class accuracy"):
        label = sample["label"]
        index_label = sample["index_label"]
        class_counts[label] += 1
        device = next(model.parameters()).device
        image = preprocess_fn(sample["image"]).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize image embedding
            logits = 100.0 * image_features @ text_features.T  # Compute similarity scores
            probs = logits.softmax(dim=-1)
            pred_index = probs[0].argmax().item()
            if pred_index == index_label:
                class_correct[label] += 1

    return {label: class_correct[label] / class_counts[label] for label in class_labels if class_counts[label] > 0}

def generate_hybrid_few_shot_features(class_labels, image_features_per_class, tokenizer, model, device, k=3, alpha=0.5):
    """Generate few-shot class features using averaged hybrid (image + text) embeddings."""
    few_shot_features = []
    for class_idx, label in enumerate(class_labels):
        if not image_features_per_class[class_idx]:
            # fallback to pure text embedding
            token = tokenizer(f"a photo of a {label}").to(device)
            with torch.no_grad():
                text_emb = model.encode_text(token)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            few_shot_features.append(text_emb.mean(dim=0))
            continue

        support_images = image_features_per_class[class_idx][:k]
        hybrid_embeds = []
        text_tokens = tokenizer([f"a photo of a {label}"] * len(support_images)).to(device)
        for j, image_feat in enumerate(support_images):
            image_feat = image_feat.to(device)
            with torch.no_grad():
                text_emb = model.encode_text(text_tokens[j].unsqueeze(0)).squeeze(0)
            hybrid = alpha * text_emb + (1 - alpha) * image_feat
            hybrid_embeds.append(hybrid)
        torch.cuda.empty_cache()
        avg_hybrid = torch.stack(hybrid_embeds).mean(dim=0)
        few_shot_features.append(avg_hybrid)
    return torch.stack(few_shot_features)

def compute_per_cluster_accuracy(class_labels, cluster_ids, per_class_accuracy):
    """Log average accuracy per semantic cluster."""
    from collections import defaultdict
    cluster_to_labels = defaultdict(list)
    for label, cluster_id in zip(class_labels, cluster_ids):
        cluster_to_labels[cluster_id].append(label)

    cluster_accuracy = {}
    for cluster_id, labels in cluster_to_labels.items():
        accs = [per_class_accuracy.get(label, 0.0) for label in labels]
        cluster_accuracy[f"cluster_{cluster_id}"] = sum(accs) / len(accs)

    wandb.log({"per_cluster_accuracy": cluster_accuracy})

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='zero-shot', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
parser.add_argument('--grayscale', type=bool, default=False, help='Grayscale images')
parser.add_argument('--transform', type=str, default=None, help='Optional transform type (e.g., "grayscale")')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (e.g., 0, 4, 6)')

args = parser.parse_args()

##===== MODEL CONFIGURATION =====##
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
tokenizer = open_clip.get_tokenizer(args.clip_model)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
##===== END OF MODEL CONFIGURATION =====##

##==== IMAGE PREPROCESSING ====##
if args.dataset == 'birdsnap':
    ds = Dataset.from_file("../birdsnap_dataset/train/data-00001-of-00139.arrow")
    json_contents = json.load(open("./birdsnap_prompts.json"))
    ds, classes_to_index, index_to_classes, captions = process_birds(ds, json_contents)
elif args.dataset == 'imagenet':
    ds = load_dataset("songweig/imagenet_sketch", trust_remote_code=True)
    ds = ds["train"]

    json_contents = json.load(open("./imagenet_prompts.json"))
    transform_type = "grayscale" if args.grayscale else None
    ds, classes_to_index, index_to_classes, captions = process_imagenet(ds, json_contents, transform=transform_type)
elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        ds = load_dataset("cifar100", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(300))  # reduced from 1000 to 300 to lower GPU memory load
        json_contents = json.load(open("./cifar100_prompts.json"))
        ds, classes_to_index, index_to_classes, captions = process_cifar100(ds, json_contents, transform=args.transform)

class_labels = [index_to_classes[i] for i in range(len(index_to_classes))]

label_prompts = [f"a photo of a {label}" for label in class_labels]
token_inputs = tokenizer(label_prompts)  # returns LongTensor
token_inputs = token_inputs.to(model.visual.conv1.weight.device)
wandb.init()
with torch.no_grad():
    label_embeddings = model.encode_text(token_inputs)
    label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)

    compute_dendrogram(label_embeddings.cpu().numpy(), class_labels)
    cluster_ids = perform_clustering(label_embeddings.cpu().numpy(), n_clusters=5)
    label_groups = [f"cluster_{cid}" for cid in cluster_ids]

# Step 1: Encode all images and collect image features grouped by class
image_features_per_class = defaultdict(list)
for sample in tqdm(ds, desc="Collecting image features per class"):
    image = preprocess_val(sample["image"]).unsqueeze(0).to(device)
    index_label = sample["index_label"]
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_feature = model.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)  # Normalize image embedding
        image_features_per_class[index_label].append(image_feature.squeeze(0).to(device))

# Step 2: Compute average image embedding per class for visualization and similarity
averaged_image_embeddings = []
image_labels = []
for idx in range(len(class_labels)):
    if image_features_per_class[idx]:
        avg_feat = torch.stack(image_features_per_class[idx]).mean(dim=0).to(device)
        averaged_image_embeddings.append(avg_feat)
        image_labels.append(index_to_classes[idx])
    else:
        averaged_image_embeddings.append(torch.zeros_like(label_embeddings[0]))
        image_labels.append(index_to_classes[idx])

image_embeddings = torch.stack(averaged_image_embeddings).cpu().numpy()

# Step 3: Generate hybrid few-shot features
few_shot_features = generate_hybrid_few_shot_features(
    class_labels, image_features_per_class, tokenizer, model, device, k=3, alpha=0.5
)
text_features = few_shot_features / few_shot_features.norm(dim=-1, keepdim=True)

# Log the 3 nearest label neighbors using text embedding similarity (for reference)
for i in tqdm(range(min(5, len(class_labels))), desc="Logging nearest neighbors"):
    label = class_labels[i]
    sims = cosine_similarity(label_embeddings[i:i+1].cpu(), label_embeddings.cpu())[0]
    top_indices = sims.argsort()[-4:-1][::-1]  # Top 3 neighbors
    neighbor_labels = [class_labels[j] for j in top_indices]
    wandb.log({f"{label}_neighbors": neighbor_labels})

# if args.grayscale:
#    print("Converting images to grayscale")
#    ds = ds.map(lambda x: {"image": x["image"].convert("L")})

##==== END OF IMAGE PREPROCESSING ====##

# Quantization: optionally add this before evaluation:

##==== WANDB CONFIGURATION ====##
wandb.init(project=args.wandb_project, config=args)

if len(ds) >= 5:
    images = [wandb.Image(ds[i]["image"], caption=f"Label: {ds[i]['label']}") for i in range(5)]
    wandb.log({"grayscale_images": images})
else:
    print("Dataset has fewer than 5 samples; skipping grayscale image logging.")

##==== WANDB CONFIGURATION END ====##

# Step 4: Evaluate each sample using image-to-text similarity with few-shot prompts
correct_counter = 0
total_counter = 0
loss = 0

batch_images = [preprocess_val(s["image"]) for s in ds]
batch_images = torch.stack(batch_images).to(device=device)

with torch.no_grad():
    for start in tqdm(range(0, len(ds), 16), desc="Evaluating samples in batches"):
        end = min(start + 16, len(ds))
        batch = ds[start:end]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        images = torch.stack([preprocess_val(s["image"]) for s in batch]).to(device)
        labels = torch.tensor([s["index_label"] for s in batch]).to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        text_probs = logits.softmax(dim=-1)
        pred_indices = text_probs.argmax(dim=-1)

        for i, sample in enumerate(batch):
            is_correct = pred_indices[i].item() == sample["index_label"]
            if is_correct:
                correct_counter += 1
            else:
                top_probs, top_ids = text_probs[i].topk(5)
                print(f"Incorrectly Predicted: {index_to_classes[pred_indices[i].item()]}, Actual: {sample['label']}, Probability: {top_probs[0].item():.4f}")
                print(f"Top 5 Predictions: {[index_to_classes[i.item()] for i in top_ids]}")
                wandb.log({
                    "misclassified_sample": {
                        "predicted": index_to_classes[pred_indices[i].item()],
                        "actual": sample['label'],
                        "top_5_predictions": [index_to_classes[i.item()] for i in top_ids],
                        "probability": top_probs[0].item()
                    }
                })
            l = loss_fn(logits[i:i+1], labels[i:i+1])
            loss += l.item()
            total_counter += 1

print(f"Accuracy: {correct_counter / total_counter * 100:.2f}%")
print(f"Total samples: {total_counter}, Correct predictions: {correct_counter}")
print(f"Average Loss: {loss / total_counter:.4f}")

wandb.log({
    "accuracy": correct_counter / total_counter,
    "total_samples": total_counter,
    "correct_predictions": correct_counter,
    "average_loss": loss / total_counter
})
