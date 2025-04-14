import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
from datasets import Dataset, Image, load_dataset
import json
from data import process_birds, process_imagenet, process_cifar100
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import tempfile
##==== END OF IMPORTS ====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')
parser.add_argument('--run_name', type=str, default='', help='WandB run name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--model_save_interval', type=int, default=5, help='Model save interval')
parser.add_argument('--eval_interval', type=int, default=2, help='Training evaluation interval')
parser.add_argument('--transform', type=str, default=None, help='Grayscale images')
##===== END OF CONFIGURATION ====##

##===== FINETUNING SCRIPT =====##
class Finetuner():

    def __init__(self, model, tokenizer, preprocess_train, preprocess_val, dataset, classes_to_index, index_to_classes, captions, epochs, batch_size, wandb_project, wandb_run_name, model_save_interval, eval_interval):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.epochs = epochs
        self.model_save_interval = model_save_interval
        self.eval_interval = eval_interval

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        if wandb_run_name:
            wandb.init(project=wandb_project, name=wandb_run_name, config=args)
        else:
            wandb.init(project=wandb_project, config=args)

        images = [wandb.Image(dataset[i]["image"], caption=f"Label: {dataset[i]['label']}") for i in range(5)]
        wandb.log({"train_images": images})

        dataset = dataset.map(lambda x : {"image" : self.preprocess_train(x["image"])})
        dataset.set_format(type="torch")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Random split
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        text = []
        for class_caption in captions:
            text.append(tokenizer(class_caption).to(self.device))

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = []
            for t in text:
                t_avg = (self.model.encode_text(t).sum(dim=0) / len(t))
                text_features.append(t_avg)
            text_features = torch.stack(text_features)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_features = self.text_features.to(self.device)

    def train(self):
        '''
        Trains the model, saves checkpoints, and evaluates the model.
        '''
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler()
        print("Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            loss_avg = 0
            counter = 0
            for sample in tqdm(self.train_loader):
                images = sample["image"].to(self.device, non_blocking=True)
                index_labels = sample["index_label"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with autocast(device_type=self.device.type):
                    images_features = model.encode_image(images)
                    images_features_normalized = images_features / images_features.norm(dim=-1, keepdim=True)                    
                    logits = (100.0 * images_features_normalized @ self.text_features.T)
                
                    loss = loss_fn(logits, index_labels)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                loss_avg += loss.detach().cpu().numpy().item()
                counter += 1
            loss_avg = loss_avg / counter
            wandb.log({"epoch": epoch, "average loss": loss_avg})

            if epoch != 0 and epoch % self.model_save_interval == 0:
                self.save_checkpoint(epoch)

            if epoch % self.eval_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_counter = 0
                    for sample in tqdm(self.val_loader):
                        images = sample["image"].to(self.device, non_blocking=True)
                        index_labels = sample["index_label"].to(self.device, non_blocking=True)
                        with autocast(device_type=self.device.type):
                            images_features = model.encode_image(images)
                            images_features_normalized = images_features / images_features.norm(dim=-1, keepdim=True)                    
                            logits = (100.0 * images_features_normalized @ self.text_features.T)
                        
                            loss = loss_fn(logits, index_labels)

                        val_loss += loss.detach().cpu().numpy().item()
                        val_counter += 1
                    val_loss = val_loss / val_counter
                    wandb.log({"epoch": epoch, "val_loss": val_loss})
                    print(f"Validation loss: {val_loss}")

    # def save_checkpoint(self, epoch):
    #     '''
    #     Saves the model checkpoint.
    #     '''
    #     checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict()
    #     }, checkpoint_path)

    #     # Log checkpoint to wandb
    #     artifact = wandb.Artifact('model-checkpoints', type='model')
    #     artifact.add_file(checkpoint_path)
    #     wandb.log_artifact(artifact)

    def save_checkpoint(self, epoch):
        '''
        Saves the model checkpoint temporarily and logs to W&B.
        '''
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
            # Save checkpoint to temporary file
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, tmp.name)

            # Log checkpoint to W&B
            artifact = wandb.Artifact('model-checkpoints', type='model') # TODO: change to a more specific name
            artifact.add_file(tmp.name, name=f"checkpoint_epoch_{epoch}.pt")
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    if args.dataset == 'imagenet':
        ds = load_dataset("imagenet-1k", split="train", trust_remote_code=True).shuffle(seed=42) # TODO: change to full dataset if necessary. Or shuffle being grabbing only 1000.
        # ds = ds.shuffle(seed=42)
        ds = ds.select(range(130000))
        json_contents = json.load(open("./imagenet_prompts.json"))
        ds, classes_to_index, index_to_classes, captions = process_imagenet(ds, json_contents, transform=args.transform)
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        ds = load_dataset("cifar100", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=42)
        json_contents = json.load(open("./cifar100_prompts.json"))
        ds, classes_to_index, index_to_classes, captions = process_cifar100(ds, json_contents, transform=args.transform)

    finetuner = Finetuner(
        model=model,
        tokenizer=tokenizer,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        dataset=ds,
        classes_to_index=classes_to_index,
        index_to_classes=index_to_classes,
        captions=captions,
        epochs=args.epochs,  # Number of epochs
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        model_save_interval=args.model_save_interval,
        eval_interval=args.eval_interval
    )
    finetuner.train()