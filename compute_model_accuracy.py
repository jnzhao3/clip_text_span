import argparse
import wandb
import torch
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.cifar100_classes import cifar100_classes
from compute_text_projection import zero_shot_classifier
from utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from torchvision import transforms
from PIL import ImageOps
from torchvision.datasets import CIFAR100

def get_args_parser():
    parser  = argparse.ArgumentParser("Compute model accuracy", add_help=False)
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b88k",
        type=str,
        help="Name of pretrained model to use",
    )
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset to use for evaluation')
    parser.add_argument('--data_path', default=None, type=str, help='input path for the dataset')
    parser.add_argument('--wandb_checkpoint_list', default=[], nargs='+', help='wandb checkpoint list to load')
    parser.add_argument('--checkpoint_epoch_list', default=[], nargs='+', help='wandb project name list')
    parser.add_argument('--transform_list', default=[], nargs='+', help='transform list to use')
    parser.add_argument('--device', default='cuda', type=str, help='device to use for testing')
    
    return parser
    
    # python compute_model_accuracy.py --model ViT-B-16 --pretrained laion2b_s34b_b88k --dataset cifar100 --data_path ~/../../../data/wong.justin/openalphaproof/output_dir --wandb_checkpoint_list jnzhao3/clip-cifar-finetuning/model-checkpoints:v12 --checkpoint_epoch_list checkpoint_epoch_4.pt --transform_list gray --device cuda

def get_classifier(model, tokenizer, classes, checkpoint, epoch, device):
    # Load the checkpoint file
    run = wandb.init()
    artifact = wandb.use_artifact(checkpoint, type='model')
    artifact_dir = artifact.download()
    checkpoint = torch.load(f"{artifact_dir}/{epoch}")
    
    # Load into model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    classname = cifar100_classes
    classifier = zero_shot_classifier(model, tokenizer, classname, OPENAI_IMAGENET_TEMPLATES, device)
    return model, classifier

def checkpoint_acc(model, dataset, classifier, device):
    correct = 0
    total = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            image_embeddings = model.encode_image(images)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            predictions = (image_embeddings @ classifier).argmax(dim=-1)
            preds = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            correct += (preds == labels).sum()
            total += labels.size
    
    return correct / total


def main(args):
    assert len(args.wandb_checkpoint_list) == len(args.checkpoint_epoch_list) == len(args.transform_list), "wandb_checkpoint_list, checkpoint_epoch_list and transform_list must have the same length"
    
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = get_tokenizer(args.model)
    
    gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        preprocess
    ])
    invert = transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.invert(img)),
        preprocess
    ])
    posterize = transforms.Compose([
        transforms.Lambda(lambda img: ImageOps.posterize(img, bits=2)),
        preprocess
    ])
    
    if args.dataset == 'cifar100':
        gray_data = CIFAR100(root=args.data_path, train=False, download=True, transform=gray)
        invert_data = CIFAR100(root=args.data_path, train=False, download=True, transform=invert)
        posterize_data = CIFAR100(root=args.data_path, train=False, download=True, transform=posterize)
    
    for wandb_checkpoint, checkpoint_epoch, transform in zip(args.wandb_checkpoint_list, args.checkpoint_epoch_list, args.transform_list):
        checkpoint_model, classifier = get_classifier(model, tokenizer, cifar100_classes, wandb_checkpoint, checkpoint_epoch, args.device)
        
        if transform == 'gray':
            dataset = gray_data
        elif transform == 'invert':
            dataset = invert_data
        elif transform == 'posterize':
            dataset = posterize_data

        accuracy = checkpoint_acc(checkpoint_model, dataset, classifier, args.device)
        print(f"Accuracy for checkpoint {wandb_checkpoint}-{checkpoint_epoch} with transform {transform}: {accuracy:.4f}")
        
        
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)