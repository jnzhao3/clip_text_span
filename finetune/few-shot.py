import torch
from PIL import Image
import open_clip
from datasets import Dataset, Image, load_dataset
import json
import argparse
import wandb
from tqdm import tqdm
from data import process_birds, process_imagenet
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
##===== END OF IMPORTS =====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='zero-shot', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
parser.add_argument('--grayscale', type=bool, default=False, help='Grayscale images')

args = parser.parse_args()

##===== MODEL CONFIGURATION =====##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Filter dataset to include only examples with valid labels from the prompt set
    valid_labels = set(json_contents.keys())
    ds = ds.filter(lambda x: x["label"] in valid_labels)

# Extract valid labels from dataset
unique_labels_in_ds = set(example["label"] for example in ds)
valid_labels = list(unique_labels_in_ds & set(json_contents.keys()))
if not valid_labels:
    raise ValueError("No overlapping labels between dataset and prompt set.")

class_labels = valid_labels

label_prompts = [f"a photo of a {label}" for label in class_labels]
token_inputs = tokenizer(label_prompts)  # returns LongTensor
token_inputs = token_inputs.to(model.visual.conv1.weight.device)
wandb.init()
with torch.no_grad():
    label_embeddings = model.encode_text(token_inputs)
    label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)

    # Project embeddings to 2D for visualization
    pca = PCA(n_components=2)
    label_embeddings_2d = pca.fit_transform(label_embeddings.cpu().numpy())

    # Create a W&B table for 2D embeddings
    embedding_table = wandb.Table(columns=["x", "y", "label"])
    for i, (x, y) in enumerate(label_embeddings_2d):
        embedding_table.add_data(x, y, class_labels[i])

    wandb.log({"class_embedding_projection": embedding_table})

    # Compute t-SNE projection (slower than PCA)
    tsne_perplexity = min(5, len(class_labels) - 1)
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init='random', random_state=42)
    label_embeddings_tsne = tsne.fit_transform(label_embeddings.cpu().numpy())

    # Create a W&B table for t-SNE embeddings
    tsne_table = wandb.Table(columns=["x", "y", "label"])
    for i, (x, y) in enumerate(label_embeddings_tsne):
        tsne_table.add_data(x, y, class_labels[i])

    # Log the t-SNE projection
    wandb.log({"class_embedding_tsne": tsne_table})

# Construct few-shot prompts
few_shot_captions = []
for i, label in enumerate(class_labels):
    sims = cosine_similarity(label_embeddings[i:i+1].cpu(), label_embeddings.cpu())[0]
    top_indices = sims.argsort()[-4:-1][::-1]  # 3 nearest neighbors
    few_shot = [f"a photo of a {class_labels[j]}." for j in top_indices]
    few_shot.append(f"a photo of a {label}.")
    few_shot_captions.append(" ".join(few_shot))

# Log a few class labels with their nearest neighbors
for i in range(min(5, len(class_labels))):  # Ensure we don’t go out of bounds
    label = class_labels[i]
    sims = cosine_similarity(label_embeddings[i:i+1].cpu(), label_embeddings.cpu())[0]
    top_indices = sims.argsort()[-4:-1][::-1]  # Top 3 neighbors
    neighbor_labels = [class_labels[j] for j in top_indices]
    wandb.log({f"{label}_neighbors": neighbor_labels})

if args.grayscale:
    print("Converting images to grayscale")
    ds = ds.map(lambda x: {"image": x["image"].convert("L")})

##==== END OF IMAGE PREPROCESSING ====##

##==== WANDB CONFIGURATION ====##
wandb.init(project=args.wandb_project, config=args)

if len(ds) >= 5:
    images = [wandb.Image(ds[i]["image"], caption=f"Label: {ds[i]['label']}") for i in range(5)]
    wandb.log({"grayscale_images": images})
else:
    print("Dataset has fewer than 5 samples; skipping grayscale image logging.")

##==== WANDB CONFIGURATION END ====##

text = []
for class_caption in few_shot_captions:
    text.append(tokenizer(class_caption).to(device=device))

with torch.no_grad(), torch.cuda.amp.autocast():
    correct_counter = 0
    total_counter = 0
    loss = 0
    text_features = []
    for t in tqdm(text):
        t_avg = (model.encode_text(t).sum(dim=0) / len(t))
        text_features.append(t_avg)

    text_features = torch.stack(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for sample in ds:
        image = sample["image"]
        label = sample["label"]
        index_label = class_labels.index(label)
        image = preprocess_val(image).unsqueeze(0).to(device=device)
        
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T)
        text_probs = logits.softmax(dim=-1)
        max_prob, index = text_probs[0].max(dim=-1)
        # grab top 5
        k = min(5, text_probs.shape[-1])
        top_five_probs, top_five_indices = text_probs[0].topk(k)
        index = index.item()
        is_correct = index == index_label
        l = loss_fn(logits, torch.tensor([index_label]).to(device=device))
        loss += l.item()
        if is_correct:
            correct_counter += 1
        else:
            print(f"Incorrectly Predicted: {index_to_classes[index]}, Actual: {label}, Probability: {max_prob.item():.4f}")
            print(f"Top 5 Predictions: {[index_to_classes[i] for i in top_five_indices.tolist()]}")
        total_counter += 1

    print(f"Accuracy: {correct_counter / total_counter * 100:.2f}%")
    print(f"Total samples: {total_counter}, Correct predictions: {correct_counter}")
    print(f"Average Loss: {loss / total_counter:.4f}")