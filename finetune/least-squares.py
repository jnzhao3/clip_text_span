import test
import torch
from PIL import Image
import open_clip
from datasets import Dataset, Image, load_dataset
import json
import argparse
import wandb
from tqdm import tqdm
# import torchvision.transforms as transforms
from torchvision import transforms
from data import process_birds, process_imagenet, process_cifar100, process_cifarc
from torch import nn
from modules import ScaledMultiheadAttention, wrap_multihead_attention
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

##===== END OF IMPORTS =====##

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

##===== MODEL CONFIGURATION =====##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
tokenizer = open_clip.get_tokenizer(args.clip_model)

if args.wrap:
    model = wrap_multihead_attention(model)

if args.checkpoint:
    run = wandb.init()
    artifact = wandb.use_artifact(args.checkpoint, type='model')
    artifact_dir = artifact.download()
    checkpoint = torch.load(f"{artifact_dir}/checkpoint_epoch_{args.checkpoint_epoch}.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = torch.optim.Adam(model.parameters())
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
        train_ds = load_dataset("cifar100", split="train", trust_remote_code=True)
        train_ds = train_ds.shuffle(seed=42)
        train_ds = train_ds.select(range(1000))
        test_ds = load_dataset("cifar100", split="test", trust_remote_code=True)
        test_ds = test_ds.shuffle(seed=42)
        test_ds = test_ds.select(range(1000))
        json_contents = json.load(open("./cifar100_prompts.json"))
        train_ds, classes_to_index, index_to_classes, captions = process_cifar100(train_ds, json_contents, preprocess_val, transform=args.transform, semantic_shift=args.semantic_shift, semantic_shuffle=args.semantic_shuffle, shift_shuffle=args.shift_shuffle)
        test_ds, _, _, _ = process_cifar100(test_ds, json_contents, preprocess_val, transform=args.transform, semantic_shift=args.semantic_shift, semantic_shuffle=args.semantic_shuffle, shift_shuffle=args.shift_shuffle)
# elif args.dataset == 'cifarc':
#     ds = load_dataset("randall-lab/cifar100-c", split="test", trust_remote_code=True)
#     ds = ds.shuffle(seed=42)
#     ds = ds.select(range(1000))
#     json_contents = json.load(open("./cifar100_prompts.json"))
#     ds, classes_to_index, index_to_classes, captions = process_cifarc(ds, json_contents, preprocess_val, transform=args.transform, semantic_shift=args.semantic_shift, semantic_shuffle=args.semantic_shuffle, shift_shuffle=args.shift_shuffle)
#     # "randall-lab/cifar100-c", split="test", trust_remote_code=True

##==== END OF IMAGE PREPROCESSING ====##

##==== WANDB CONFIGURATION ====##
wandb.init(project=args.wandb_project, config=args)

images = [wandb.Image(train_ds[i]["image"], caption=f"Label: {index_to_classes[train_ds[i]['index_label']]}") for i in range(5)]
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

##==== LEARN LEAST SQUARES =====##
train_dataloader = DataLoader(train_ds, batch_size=10000, shuffle=True)

# for sample in train_dataloader:
with torch.no_grad(), autocast(device_type=device.type):
    sample = next(iter(train_dataloader))
    # index_label = sample["index_label"]
    # image = image.unsqueeze(0).to(device=device)
    # transformed_image = image.unsqueeze(0).to(device=device)[0]
    # transformed_image = transformed_image.unsqueeze(0).to(device=device)
    image = sample["image"].to(device, dtype=torch.float16, non_blocking=True)
    transformed_image = transforms.Grayscale(num_output_channels=3)(image).to(device, dtype=torch.float16, non_blocking=True)

    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    transformed_image_features = model.encode_image(transformed_image)
    transformed_image_features /= transformed_image_features.norm(dim=-1, keepdim=True)

    ones = torch.ones(transformed_image_features.size(0), 1, dtype=transformed_image_features.dtype, device=device)
    X_aug = torch.cat([transformed_image_features, ones], dim=1)

    # cast to float32 for lstsq
    X_aug = X_aug.to(torch.float32)
    image_features = image_features.to(torch.float32)
    W = torch.linalg.lstsq(X_aug, image_features).solution

    A = W[:-1, :]
    b = W[-1, :]

    # loss = transformed_image_features @ A.T + b
    # loss = nn.functional.mse_loss(loss, image_features)

    # import ipdb; ipdb.set_trace()

    print("finished calculating least squares")

###==== END OF LEARN LEAST SQUARES =====##


with torch.no_grad(), autocast(device_type=device.type):
    for sample in train_ds: # TODO: change this back
        index_label = sample["index_label"]
        image = sample["image"]
        transformed_image = transforms.Grayscale(num_output_channels=3)(image).to(device=device)
        transformed_image = transformed_image.unsqueeze(0).to(device=device)

        transformed_image_features = model.encode_image(transformed_image)
        transformed_image_features /= transformed_image_features.norm(dim=-1, keepdim=True)
        # transformed_image_features = transformed_image_features.to(torch.float32)
        # transformed_image_features_ls = transformed_image_features @ A + b
        # transformed_image_features_ls /= transformed_image_features_ls.norm(dim=-1, keepdim=True)
        
        transformed_image_features_ls = transformed_image_features + diff

        image = image.unsqueeze(0).to(device=device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * transformed_image_features_ls @ text_features.T)
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
            print(f"Incorrectly Predicted: {index_to_classes[index]}, Actual: {index_to_classes[index_label]}, Probability: {max_prob.item():.4f}")
            print(f"Top 5 Predictions: {[index_to_classes[i] for i in top_five_indices.tolist()]}")
        total_counter += 1

    print(f"Accuracy: {correct_counter / total_counter * 100:.2f}%")
    print(f"Total samples: {total_counter}, Correct predictions: {correct_counter}")
    print(f"Average Loss: {loss / total_counter:.4f}")