# import torch
# from torchvision import datasets, transforms

# # Define a transform for image normalization
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# # Load CIFAR-10 dataset
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# # Load data using DataLoader
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


from datasets import load_dataset

ds = load_dataset("sasha/birdsnap")

import ipdb; ipdb.set_trace()