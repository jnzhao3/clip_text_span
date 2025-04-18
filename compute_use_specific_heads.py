import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import random
import tqdm
from utils.misc import accuracy
from torchvision.datasets import CIFAR100


def full_accuracy(preds, labels, locs_attributes):
    locs_labels = labels.detach().cpu().numpy()
    accs = {}
    for i in [0, 1]:
        for j in [0, 1]:
            locs = np.logical_and(locs_labels == i, locs_attributes == j)
            accs[f"({i}, {j})"] = accuracy(preds[locs], labels[locs])[0] * 100
    accs[f"full"] = accuracy(preds, labels)[0] * 100
    return accs


def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--figures_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary_waterbirds",
        help="imagenet, waterbirds, waterbirds_binary or cub",
    )
    parser.add_argument("--data_path", default="", type=str, help="dataset path")
    return parser


def main(args):
    if args.model == "ViT-H-14":
        to_mean_ablate_setting = [(31, 12), (30, 11), (29, 4)]
        to_mean_ablate_geo = [(31, 8), (30, 15), (30, 12), (30, 6), (29, 14), (29, 8)]
    elif args.model == "ViT-L-14":
        # geo location heads
        to_mean_ablate_geo = [(21, 1), (22, 12), (22, 13), (21, 11), (21, 14), (23, 6)]
        to_mean_ablate_setting = [(21, 3), (21, 6), (21, 8), (21, 13), (22, 2), (22, 12), (22, 15), (23, 1), (23, 3), (23, 5)]
        to_mean_ablate_color = [(21, 0), (21, 9), (22, 10), (22, 11), (23, 8)]
    elif args.model == "ViT-B-16":
        to_mean_ablate_setting = [(11, 3), (10, 11), (10, 10), (9, 8), (9, 6)]
        to_mean_ablate_geo = [(11, 6), (11, 0)]
    else:
        raise ValueError('model not analyzed')
    # to_mean_ablate_output = to_mean_ablate_geo + to_mean_ablate_setting
    to_mean_ablate_output = to_mean_ablate_color
    original_dataset = args.dataset.split("_")[0] if "gray" in args.dataset else args.dataset
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), "rb"
    ) as f:
        mlps = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{original_dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)

    if args.dataset == "imagenet":
        labels = np.array([i // 50 for i in range(attns.shape[0])])
    elif "CIFAR100" in args.dataset:
        ds = CIFAR100(root=args.data_path, download=True, train=False)
        labels = np.array([ds.targets[i] for i in range(attns.shape[0])])
    else:
        with open(
            os.path.join(args.input_dir, f"{args.dataset}_labels.npy"), "rb"
        ) as f:
            labels = np.load(f)
            labels = labels[:, :, 0]
            
    baseline = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    # baseline_acc = full_accuracy(
    #     torch.from_numpy(baseline @ classifier).float(),
    #     torch.from_numpy(labels[:, 0]),
    #     labels[:, 1],
    # )
    baseline_acc = accuracy(torch.from_numpy(baseline @ classifier).float(), torch.from_numpy(labels))
    
    print("Baseline:", baseline_acc)
    for layer, head in to_mean_ablate_output:
        attns[:, layer, head, :] = np.mean(
            attns[:, layer, head, :], axis=0, keepdims=True
        )
    ablated = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    ablated_acc = accuracy(torch.from_numpy(ablated @ classifier).float(), torch.from_numpy(labels))
    print("color heads mean ablated:", ablated_acc)
    
    for layer, head in to_mean_ablate_output:
        zero_ablated_attns = attns.copy()
        zero_ablated_attns[:, layer, head, :] = 0
    ablated = zero_ablated_attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    ablated_acc = accuracy(torch.from_numpy(ablated @ classifier).float(), torch.from_numpy(labels))
    print("color heads zero ablated:", ablated_acc)
    
    for layer in range(attns.shape[1] - 4):
        for head in range(attns.shape[2]):
            attns[:, layer, head, :] = np.mean(
                attns[:, layer, head, :], axis=0, keepdims=True
            )
    ablated = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    ablated_acc = accuracy(torch.from_numpy(ablated @ classifier).float(), torch.from_numpy(labels))
    print("first few attention layers + color heads mean ablated:", ablated_acc)
    
    for layer in range(mlps.shape[1]):
        mlps[:, layer] = np.mean(mlps[:, layer], axis=0, keepdims=True)
    ablated = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    # ablated_acc = full_accuracy(
    #     torch.from_numpy(ablated @ classifier).float(),
    #     torch.from_numpy(labels[:, 0]),
    #     labels[:, 1],
    # )
    ablated_acc = accuracy(torch.from_numpy(ablated @ classifier).float(), torch.from_numpy(labels))
    print("globally mean ablated:", ablated_acc)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)
