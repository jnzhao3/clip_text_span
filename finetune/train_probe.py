#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import open_clip
from datasets import Dataset, load_dataset, Image as HFImage
import json
import argparse
import wandb
from tqdm import tqdm

from data import process_birds, process_imagenet

# Global dictionary to store intermediate activations.
activation = {}

def get_head_activation(name, target_head, head_dim):
    """
    Returns a hook function that captures the output for a specific attention head.
    Assumes that the module's output is of shape [B, tokens, embed_dim] with concatenated heads.
    """
    def hook(model, input, output):
        # output shape: [B, tokens, embed_dim]
        # Extract channels corresponding to the target attention head.
        head_output = output[..., target_head*head_dim:(target_head+1)*head_dim]
        activation[name] = head_output.detach()
    return hook

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='zero-shot', help='WandB project name')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name (e.g., birdsnap or imagenet)')
    parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
    parser.add_argument('--grayscale', type=bool, default=False, help='Convert images to grayscale')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of probe training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for probe training')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the CLIP model and its transforms.
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.to(device)
    model.eval()  # Make sure CLIP is in evaluation mode.
    
    # Freeze the CLIP model parameters so that only the probe will be updated.
    for param in model.parameters():
        param.requires_grad = False

    # ===== Load and Process Dataset =====
    if args.dataset == 'birdsnap':
        ds = Dataset.from_file("../birdsnap_dataset/train/data-00001-of-00139.arrow")
        with open("./birdsnap_prompts.json", "r") as f:
            json_contents = json.load(f)
        ds, classes_to_index, index_to_classes, captions = process_birds(ds, json_contents)
    elif args.dataset == 'imagenet':
        ds = load_dataset("imagenet-1k", split="train[:1000]", trust_remote_code=True)
        with open("./imagenet_prompts.json", "r") as f:
            json_contents = json.load(f)
        ds, classes_to_index, index_to_classes, captions = process_imagenet(ds, json_contents)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.grayscale:
        print("Converting images to grayscale")
        ds = ds.map(lambda x: {"image": x["image"].convert("L")})

    # Log a few sample images to wandb.
    sample_images = [wandb.Image(ds[i]["image"], caption=f"Label: {ds[i]['label']}") for i in range(min(5, len(ds)))]
    wandb.log({"sample_images": sample_images})

    # ===== Register Forward Hook for Layer 22, Attention Head 10 =====
    # Here we assume that CLIP's visual transformer blocks are stored in resblocks.
    # Using a zero-index, layer 22 corresponds to index 21.
    target_layer = 21  # Change this value if your model indexing differs.
    target_head = 10   # The desired attention head index.
    
    # Access the target attention module.
    attention_module = model.visual.transformer.resblocks[target_layer].attn
    # Assume that the attention module has a 'num_heads' attribute.
    num_heads = attention_module.num_heads
    # Compute the per-head dimension from the visual embedding dimension.
    head_dim = model.visual.embed_dim // num_heads

    # Register the hook that saves only the specified attention head's activations.
    attention_module.register_forward_hook(get_head_activation('layer22_head10', target_head, head_dim))

    # ===== Obtain Dummy Activation to Determine Dimensions =====
    dummy_image = ds[0]["image"]
    image_tensor = preprocess_val(dummy_image).unsqueeze(0).to(device)
    _ = model.encode_image(image_tensor)
    act = activation['layer22_head10']
    # If the activation still has spatial dimensions, pool them (e.g. global average pooling).
    if act.ndim == 4:
        act = act.mean(dim=[2, 3])
    input_dim = act.shape[1]
    
    # Get the target CLIP image embedding dimension.
    with torch.no_grad():
        target_embedding = model.encode_image(image_tensor)
    output_dim = target_embedding.shape[1]

    print(f"Probe input dimension: {input_dim}, output dimension: {output_dim}")

    # ===== Create the Probe Network =====
    probe = LinearProbe(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    # ===== Training Loop for the Probe =====
    num_epochs = args.num_epochs
    probe.train()
    print("Starting probe training on activations from layer 22 attention head 10...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0
        for sample in tqdm(ds, desc=f"Epoch {epoch+1}/{num_epochs}"):
            image = sample["image"]
            image_tensor = preprocess_val(image).unsqueeze(0).to(device)

            optimizer.zero_grad()

            # Forward pass through the CLIP model.
            # This call returns the final CLIP image embedding and triggers our hook.
            output = model.encode_image(image_tensor)
            target = output.detach()  # Ground truth CLIP embedding.

            # Retrieve the specific intermediate activation.
            act = activation['layer22_head10']
            if act.ndim == 4:
                act = act.mean(dim=[2, 3])
            # The activation should now be of shape [1, input_dim].

            # Use the probe to predict the final CLIP embedding.
            pred = probe(act)

            # (Optional) Add normalization here if desired.
            loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss})

    print("Probe training completed.")

if __name__ == "__main__":
    main()
