import torch
from PIL import Image
import open_clip
from datasets import Dataset, Image, load_dataset
import json
import argparse
import wandb
from tqdm import tqdm
import torchvision.transforms as transforms
from data import process_birds, process_imagenet, process_cifar100
from torch import nn
##===== END OF IMPORTS =====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='zero-shot', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
parser.add_argument('--transform', type=str, default=None, help='Grayscale images')

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
elif args.dataset == 'imagenet_sketch':
    ds = load_dataset("imagenet_sketch", split="train", trust_remote_code=True)
    ds = ds.select(range(1000))
    json_contents = json.load(open("./imagenet_prompts.json"))
    ds, classes_to_index, index_to_classes, captions = process_imagenet(ds, json_contents, transform=args.transform)
elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        ds = load_dataset("cifar100", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(1000))
        json_contents = json.load(open("./cifar100_prompts.json"))
        ds, classes_to_index, index_to_classes, captions = process_cifar100(ds, json_contents, transform=args.transform)

# if args.grayscale:
#     print("Converting images to grayscale")
#     ds = ds.map(lambda x: {"image": x["image"].convert("L")})

##==== END OF IMAGE PREPROCESSING ====##

##==== WANDB CONFIGURATION ====##
wandb.init(project=args.wandb_project, config=args)

images = [wandb.Image(ds[i]["image"], caption=f"Label: {ds[i]['label']}") for i in range(5)]
wandb.log({"images": images})

##==== WANDB CONFIGURATION END ====##

text = []
for class_caption in captions:
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
        index_label = sample["index_label"]
        image = preprocess_val(image).unsqueeze(0).to(device=device)
        
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T)
        text_probs = logits.softmax(dim=-1)
        max_prob, index = text_probs[0].max(dim=-1)
        # grab top 5
        top_five_probs, top_five_indices = text_probs[0].topk(5)
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