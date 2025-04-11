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
from data import process_birds, process_imagenet
from tqdm import tqdm
from torch.amp import autocast, GradScaler
##==== END OF IMPORTS ====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')
parser.add_argument('--dataset', type=str, default='birdsnap', help='Dataset name')
parser.add_argument('--clip_model', type=str, default='hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', help='CLIP model name')

##===== FINETUNING SCRIPT =====##
class Finetuner():

    def __init__(self, model, tokenizer, preprocess_train, preprocess_val, dataset, classes_to_index, index_to_classes, captions, epochs, batch_size, wandb_project):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.epochs = epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
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
        # text = self.tokenizer(self.captions).to(self.device)
        # with torch.no_grad(), torch.cuda.amp.autocast():
        #     text_features = self.model.encode_text(text)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler()
        print("Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            loss_avg = 0
            counter = 0
            for sample in tqdm(self.train_loader):
                images = sample["image"].to(self.device)
                labels = sample["label"]
                index_labels = sample["index_label"].to(self.device)
                self.optimizer.zero_grad()
                with torch.no_grad(), autocast(device_type=self.device.type):
                    images_features = model.encode_image(images)
                    images_features_normalized = images_features / images_features.norm(dim=-1, keepdim=True)
                    print("Calculated image features")
                    
                    logits = (100.0 * images_features_normalized @ self.text_features.T)
                    print("Calculated logits")
                
                    loss = loss_fn(logits, index_labels)

                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                # loss.backward()
                # self.optimizer.step()

                # loss_avg += loss.item()
                loss_avg += loss.detach().cpu().numpy().item()
                counter += 1
            loss_avg = loss_avg / counter
            wandb.log({"epoch": epoch, "average loss": loss_avg})

            # Validation step
            # self.validate(self.val_loader)
            # Save checkpoint
            # self.save_checkpoint(epoch)

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
    args = parser.parse_args()
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.clip_model)
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    if args.dataset == 'imagenet':
        ds = load_dataset("imagenet-1k", split="train[:1000]", trust_remote_code=True)
        # ds = ds["train"].select(range(20))
        json_contents = json.load(open("./imagenet_prompts.json"))
        ds, classes_to_index, index_to_classes, captions = process_imagenet(ds, json_contents)

    # dataset = Dataset.from_file("../birdsnap_dataset/train/data-00003-of-00139.arrow")
    # # Cast the image column to the Image feature
    # dataset = dataset.cast_column("image", Image())

    # Load the JSON file
    # json_contents = json.load(open("./birdsnap_prompts.json"))
    # classes = json_contents["classes"]
    # templates = json_contents["templates"]

    finetuner = Finetuner(
        model=model,
        tokenizer=tokenizer,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        dataset=ds,
        classes_to_index=classes_to_index,
        index_to_classes=index_to_classes,
        captions=captions,
        epochs=10,  # Number of epochs
        batch_size=4,  # Batch size
        wandb_project=args.wandb_project
    )
    finetuner.train()