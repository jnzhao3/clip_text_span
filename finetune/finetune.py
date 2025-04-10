import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
##==== END OF IMPORTS ====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')

##===== FINETUNING SCRIPT =====##
class Finetuner():

    def __init__(self, model, tokenizer, preprocess_train, preprocess_val, epochs, batch_size, wandb_project):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        wandb.init(project=wandb_project, config=args)

        # Download the training set
        # train_dataset = datasets.ImageNet(root='path_to_imagenet_data', split='train', transform=self.preprocess_train)
        # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.preprocess_train)
#         coco_dataset = datasets.CocoDetection(
#     root='./data/train2017',
#     annFile='./data/annotations/instances_train2017.json',
#     transform=transform
# # )
        # train_dataset = datasets.CocoDetection(root='./data/train2017', annFile='./data/annotations/instances_train2017.json', transform=self.preprocess_train)
        train_dataset = datasets.ImageFolder(root='./country211/train', transform=self.preprocess_train)

        # Download the validation set
        # val_dataset = datasets.ImageNet(root='path_to_imagenet_data', split='val', transform=self.preprocess_val)
        # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.preprocess_val)
        # val_dataset = datasets.CocoDetection(root='./data/val2017', annFile='./data/annotations/instances_val2017.json', transform=self.preprocess_val)
        val_dataset = datasets.ImageFolder(root='./country211/valid', transform=self.preprocess_val)
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for images, texts in self.train_loader:
                import ipdb; ipdb.set_trace()
                images = images.to(self.device)
                texts = [self.tokenizer(text).to(self.device) for text in texts]

                self.optimizer.zero_grad()
                logits_per_image, logits_per_text = self.model(images, texts)
                loss = F.cross_entropy(logits_per_image, torch.arange(len(images)).to(self.device))
                loss.backward()
                self.optimizer.step()

            wandb.log({"epoch": epoch, "loss": loss.item()})

            # Validation step
            self.validate(self.val_loader)
            # Save checkpoint
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)

        # Log checkpoint to wandb
        artifact = wandb.Artifact('model-checkpoints', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

if __name__ == "__main__":

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

    args = parser.parse_args()

    finetuner = Finetuner(
        model=model,
        tokenizer=tokenizer,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        epochs=10,  # Number of epochs
        batch_size=32,  # Batch size
        wandb_project=args.wandb_project
    )
    finetuner.train()