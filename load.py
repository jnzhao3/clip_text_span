import wandb
import torch
import open_clip

run = wandb.init()

# Load the artifact
artifact = wandb.use_artifact('jnzhao3/clip-finetuning/model-checkpoints:v0', type='model')
artifact_dir = artifact.download()

# Load the checkpoint file
checkpoint = torch.load(f"{artifact_dir}/checkpoint_epoch_0.pt")

# Load into model

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

model.load_state_dict(checkpoint['model_state_dict'])

# Optionally restore optimizer state
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
