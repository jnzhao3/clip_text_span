import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
from PIL import Image
import open_clip
from datasets import Dataset, Image, load_dataset
import json
import wandb
from tqdm import tqdm
from data import process_birds, process_imagenet
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import argparse


def compute_dendrogram(embeddings, labels):
    linkage_matrix = sch.linkage(embeddings, method='ward')
    plt.figure(figsize=(12, 6))
    dendro = sch.dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title("Dendrogram of Label Embeddings")
    plt.xlabel("Class Label")
    plt.ylabel("Distance")
    wandb.log({"Dendrogram": wandb.Image(plt)})
    plt.close()

def perform_clustering(embeddings, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return clustering.fit_predict(embeddings)

def compute_tsne(embeddings, labels, groups):
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
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            pred_index = probs[0].argmax().item()
            if pred_index == index_label:
                class_correct[label] += 1
    return {label: class_correct[label] / class_counts[label] for label in class_labels if class_counts[label] > 0}

def compute_per_cluster_accuracy(class_labels, cluster_ids, per_class_accuracy):
    cluster_to_labels = defaultdict(list)
    for label, cluster_id in zip(class_labels, cluster_ids):
        cluster_to_labels[cluster_id].append(label)
    cluster_accuracy = {}
    for cluster_id, labels in cluster_to_labels.items():
        accs = [per_class_accuracy.get(label, 0.0) for label in labels]
        cluster_accuracy[f"cluster_{cluster_id}"] = sum(accs) / len(accs)
    wandb.log({"per_cluster_accuracy": cluster_accuracy})

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_ddp(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        wandb.init(project=args.wandb_project, config=args)

    # === Load dataset ===
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

    class_labels = [index_to_classes[i] for i in range(len(index_to_classes))]

    label_prompts = [f"a photo of a {label}" for label in class_labels]
    token_inputs = tokenizer(label_prompts).to(device)
    with torch.no_grad():
        label_embeddings = model.module.encode_text(token_inputs)
        label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)

    if rank == 0:
        compute_dendrogram(label_embeddings.cpu().numpy(), class_labels)
        cluster_ids = perform_clustering(label_embeddings.cpu().numpy(), n_clusters=5)
        label_groups = [f"cluster_{cid}" for cid in cluster_ids]
        label_embeddings_2d = compute_pca(label_embeddings.cpu().numpy(), class_labels, label_groups)
        label_embeddings_tsne = compute_tsne(label_embeddings.cpu().numpy(), class_labels, label_groups)
    else:
        cluster_ids = perform_clustering(label_embeddings.cpu().numpy(), n_clusters=5)
        label_groups = [f"cluster_{cid}" for cid in cluster_ids]

    few_shot_captions = []
    for i, label in enumerate(class_labels):
        sims = cosine_similarity(label_embeddings[i:i+1].cpu(), label_embeddings.cpu())[0]
        top_indices = sims.argsort()[-4:-1][::-1]
        few_shot = [f"a photo of a {class_labels[j]}." for j in top_indices]
        few_shot.append(f"a photo of a {label}.")
        few_shot_captions.append(" ".join(few_shot))

    text = [tokenizer(caption).to(device) for caption in few_shot_captions]
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = []
        for t in text:
            t_avg = (model.module.encode_text(t).sum(dim=0) / len(t))
            text_features.append(t_avg)
        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        if rank == 0:
            per_class_accuracy = compute_per_class_accuracy(ds, model.module, preprocess_val, text_features, class_labels)
            wandb.log({"per_class_accuracy": per_class_accuracy})
            compute_per_cluster_accuracy(class_labels, cluster_ids, per_class_accuracy)

            correct_counter = 0
            total_counter = 0
            loss = 0

            for sample in tqdm(ds, desc="Evaluating samples"):
                image = preprocess_val(sample["image"]).unsqueeze(0).to(device)
                label = sample["label"]
                index_label = sample["index_label"]
                image_features = model.module.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ text_features.T
                text_probs = logits.softmax(dim=-1)
                max_prob, index = text_probs[0].max(dim=-1)
                k = min(5, text_probs.shape[-1])
                top_five_probs, top_five_indices = text_probs[0].topk(k)
                index = index.item()
                is_correct = index == index_label
                l = nn.CrossEntropyLoss()(logits, torch.tensor([index_label]).to(device))
                loss += l.item()
                if is_correct:
                    correct_counter += 1
                else:
                    wandb.log({
                        "misclassified_sample": {
                            "predicted": index_to_classes[index],
                            "actual": label,
                            "top_5_predictions": [index_to_classes[i] for i in top_five_indices.tolist()],
                            "probability": max_prob.item()
                        }
                    })
                total_counter += 1

            wandb.log({
                "accuracy": correct_counter / total_counter,
                "total_samples": total_counter,
                "correct_predictions": correct_counter,
                "average_loss": loss / total_counter
            })

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='zero-shot')
    parser.add_argument('--dataset', type=str, default='birdsnap')
    parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    parser.add_argument('--grayscale', type=bool, default=False)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main_ddp, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()