import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import argparse
##==== END OF IMPORTS ====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

##===== FINETUNING SCRIPT =====##
def finetune():
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    # Initialize Weights and Biases
    wandb.init(project=args.wandb_project, config=args)
    # Store model weights in wandb
    wandb.watch(model, log="all")

    epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

    # Log checkpoint to wandb
    artifact = wandb.Artifact('model-checkpoints', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
