import os
import numpy as np
import argparse
from torchvision.datasets import CIFAR100
from utils.misc import accuracy
import torch

def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model_A",
        default="ViT-B-16",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument(
        "--model_B",
        default="ViT-B-16",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--dataset_A",
        type=str,
        default="CIFAR100",
        help="dataset to use for evaluation",
    )
    parser.add_argument(
        "--dataset_B",
        type=str,
        default="CIFAR100",
        help="dataset to use for evaluation",
    )
    parser.add_argument("--data_path", default="", type=str, help="dataset path")
    return parser

def main(args):
    with open(
        os.path.join(args.input_dir, f"{args.dataset_A}_attn_{args.model_A}.npy"), "rb"
    ) as f:
        attns_a = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset_B}_attn_{args.model_B}.npy"), "rb"
    ) as f:
        attns_b = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset_A}_mlp_{args.model_A}.npy"), "rb"
    ) as f:
        mlps_a = np.load(f)  # [b, l+1, d]  
    with open(
        os.path.join(args.input_dir, f"{args.dataset_B}_mlp_{args.model_B}.npy"), "rb"
    ) as f:
        mlps_b = np.load(f)  # [b, l+1, d]
    
    # gray for now, need to change to accept genral transformation later
    original_dataset = args.dataset_A.split("_")[0] if "gray" in args.dataset_A else args.dataset_A
    with open(
        os.path.join(args.input_dir, f"{original_dataset}_classifier_{args.model_A}.npy"), "rb"
    ) as f:
        classifier_a = np.load(f)
    with open(
        os.path.join(args.input_dir, f"{original_dataset}_classifier_{args.model_B}.npy"), "rb"
    ) as f:
        classifier_b = np.load(f)

    if "CIFAR100" in args.dataset_A:
        ds = CIFAR100(root=args.data_path, download=True, train=False)
        labels = ds.targets
    
    if "ViT-L-14" in args.model_A:
        layers_we_care = [20, 21, 22, 23]
        heads_per_layer = list(range(16))
    elif "ViT-B-16" in args.model_A:
        layers_we_care = [8, 9, 10, 11]
        heads_per_layer = list(range(12))
    
    representation_b = attns_b.sum(axis=(1, 2)) + mlps_b.sum(axis=1)
    acc_b = accuracy(torch.from_numpy(representation_b @ classifier_b).float(), torch.tensor(labels))
    print(args.model_B, "accuracy:", acc_b)
    
    diff_per_head = {(i, j): 0 for i in layers_we_care for j in heads_per_layer}
    
    for i in layers_we_care:
        for j in heads_per_layer:
            for k in range(attns_a.shape[0]):
                diff_per_head[(i, j)] += np.linalg.norm(
                    attns_a[k, i, j, :] - attns_b[k, i, j, :]
                )
    sorted_diff = sorted(diff_per_head.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 differences per head:")
    for (layer, head), diff in sorted_diff[:10]:
        print(f"Layer {layer}, Head {head}: {diff:.4f}")
    
    representation = attns_a.sum(axis=(1, 2)) + mlps_a.sum(axis=1)
    prediction_gray = representation @ classifier_a
    prediction_gray = prediction_gray.argmax(axis=1)
    incorrect = []
    correct = []
    diff_per_head = {(i, j): 0 for i in layers_we_care for j in heads_per_layer}
    for i in range(len(prediction_gray)):
        if prediction_gray[i] != labels[i]:
            incorrect.append(i)
        else:
            correct.append(i)
            
    acc_a = len(correct) / (len(correct) + len(incorrect))
    print(args.model_A, "accuracy:", acc_a)
    
    for i in layers_we_care:
        for j in heads_per_layer:
            for k in incorrect:
                diff_per_head[(i, j)] += np.linalg.norm(
                    attns_a[k, i, j, :] - attns_b[k, i, j, :]
                )
    sorted_diff = sorted(diff_per_head.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 differences per head for incorrectly predicted ones:")
    for (layer, head), diff in sorted_diff[:10]:
        print(f"Layer {layer}, Head {head}: {diff:.4f}")
        
    diff_per_head = {(i, j): 0 for i in layers_we_care for j in heads_per_layer}    
    for i in layers_we_care:
        for j in heads_per_layer:
            for k in correct:
                diff_per_head[(i, j)] += np.linalg.norm(
                    attns_a[k, i, j, :] - attns_b[k, i, j, :]
                )
    sorted_diff = sorted(diff_per_head.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 differences per head for correctly predicted ones:")
    for (layer, head), diff in sorted_diff[:10]:
        print(f"Layer {layer}, Head {head}: {diff:.4f}")
        
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
