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
from datasets import Dataset, Image
##==== END OF IMPORTS ====##

##==== CONFIGURATION ====##
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='clip-finetuning', help='WandB project name')

##===== FINETUNING SCRIPT =====##
class Finetuner():

    def __init__(self, model, tokenizer, preprocess_train, preprocess_val, dataset, classes, templates, epochs, batch_size, wandb_project):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
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
        # train_dataset = datasets.ImageFolder(root='./country211/train', transform=self.preprocess_train)


        # Download the validation set
        # val_dataset = datasets.ImageNet(root='path_to_imagenet_data', split='val', transform=self.preprocess_val)
        # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.preprocess_val)
        # val_dataset = datasets.CocoDetection(root='./data/val2017', annFile='./data/annotations/instances_val2017.json', transform=self.preprocess_val)
        # train_datas
        # val_dataset = datasets.ImageFolder(root='./country211/valid', transform=self.preprocess_val)
        # Create data loaders
        # train_dataset = 
        # Split sizes
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Random split
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.captions = []
        for i in range(len(classes)):
            # for j in range(len(templates)):
                # captions.append(templates[j].replace("{classname}", classes[i]))
            self.captions.append(templates[0][0] + classes[i] + templates[0][1])

    def train(self):
        text = self.tokenizer(self.captions).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        for epoch in range(self.epochs):
            self.model.train()
            for sample in self.train_loader:
                import ipdb; ipdb.set_trace()
                images = sample["image"]
                labels = sample["label"]
                # images = [self.preprocess_train(image).unsqueeze(0) for image in images]
                images = self.preprocess_train(images).to(self.device)
                images_features = model.encode_image(images)
                images_features /= images_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * images_features @ text_features.T).softmax(dim=-1)
                label_one_hot
                # images = images.to(self.device)
                # texts = [self.tokenizer(text).to(self.device) for text in texts]

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

    dataset = Dataset.from_file("../birdsnap_dataset/train/data-00003-of-00139.arrow")
    # Cast the image column to the Image feature
    dataset = dataset.cast_column("image", Image())

    args = parser.parse_args()

    # Load the JSON file
    json_contents = json.load(open("./birdsnap_prompts.json"))
    classes = json_contents["classes"]
    templates = json_contents["templates"]

    finetuner = Finetuner(
        model=model,
        tokenizer=tokenizer,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        dataset=dataset,
        classes=classes,
        templates=templates,
        epochs=10,  # Number of epochs
        batch_size=32,  # Batch size
        wandb_project=args.wandb_project
    )
    finetuner.train()