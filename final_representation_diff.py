import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
from PIL import ImageOps
from utils.factory import create_model_and_transforms


with open("text_descriptions/image_descriptions_general.txt", "r") as f:
    text_description = [i.replace("\n", "") for i in f.readlines()]
    
transform = "gray"
model_name = "ViT-B-16"
pretrained = "laion2b_s34b_b88k"

original_attn_representation = np.load("output_dir/CIFAR100_attn_ViT-B-16.npy")
original_mlp_representation = np.load("output_dir/CIFAR100_mlp_ViT-B-16.npy")
original_final_representation = original_attn_representation.sum(axis=(1, 2)) + original_mlp_representation.sum(axis=1)

image_descriptions = np.load("output_dir/image_descriptions_general_ViT-B-16.npy")

if transform == "gray":
    transformed_attn_representation = np.load("output_dir/CIFAR100_gray_attn_ViT-B-16.npy")
    transformed_mlp_representation = np.load("output_dir/CIFAR100_gray_mlp_ViT-B-16.npy")
    transformed_final_representation = transformed_attn_representation.sum(axis=(1, 2)) + transformed_mlp_representation.sum(axis=1)
elif transform == "invert":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_final_representation = []
    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    ds = CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Lambda(lambda img: ImageOps.invert(img)), preprocess]))
    dataloader = DataLoader(ds, batch_size=100, shuffle=False)
    for image, label in tqdm.tqdm(dataloader):
        image = image.to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        transformed_final_representation.append(image_features.cpu().numpy())
    transformed_final_representation = np.array(transformed_final_representation)
    transformed_final_representation = np.reshape(transformed_final_representation, (original_final_representation.shape[0], -1))


diff = original_final_representation - transformed_final_representation

U, S, VT = np.linalg.svd(diff)
print("First 10 Singular Values:", S[:10])
print("SVD done!")

# plt.figure(figsize=(10, 6))
# plt.bar(range(1, 21), S[:20], color='b', alpha=0.7)
# plt.title("First 20 Singular Values")
# plt.xlabel("Index")
# plt.ylabel("Singular Value")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

approximations = []
average = []

for i in range(10):
    approximations.append(np.outer(U[:, i], VT[i, :]) * S[i])
    average.append(np.mean(approximations[i], axis=0))

# import pickle
# with open("rank_1_average.pkl", "wb") as f:
#     pickle.dump(rank_1_average, f)
# print(rank_1_average)

for i, rank_1_average in enumerate(average):
    
    similarity_scores = []
    for description in image_descriptions:
        sim = np.dot(rank_1_average, description)/(np.linalg.norm(rank_1_average) * np.linalg.norm(description))
        similarity_scores.append(sim)
        
    print(f"{i+1} singular vector:")
    
    result = zip(text_description, similarity_scores)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    for i in range(10):
        print(f"Top 10 Description: {sorted_result[i][0]}, Similarity Score: {sorted_result[i][1]}")
    for i in range(10):
        print(f"Bottom 10 Description: {sorted_result[-i-1][0]}, Similarity Score: {sorted_result[-i-1][1]}")
    print('\n')


