import torch
from PIL import Image
import open_clip
from datasets import Dataset, Image
import json
##===== END OF IMPORTS =====##

##===== MODEL CONFIGURATION =====##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

model.to(device)
##===== END OF MODEL CONFIGURATION =====##

##==== IMAGE PREPROCESSING ====##
# img_path = "../country211/test/CN/413200_30.002219_93.881607.jpg"
# Load the dataset
ds = Dataset.from_file("../birdsnap_dataset/train/data-00002-of-00139.arrow")

# Cast the image column to the Image feature
ds = ds.cast_column("image", Image())  # Replace 'image_column_name' with your actual column name

# Access an image
# image = preprocess_val(ds[0]["image"]).unsqueeze(0).cuda(device=device)
##==== END OF IMAGE PREPROCESSING ====##

##===== TEXT PREPROCESSING ====##
# image = preprocess_val(Image.open(img_path)).unsqueeze(0).cuda(device=device)
# text = tokenizer(["a tree", "a bird", "a cat"]).cuda(device=device)
json_contents = json.load(open("./birdsnap_prompts.json"))
classes = json_contents["classes"]
templates = json_contents["templates"]

captions = []
for i in range(len(classes)):
    # for j in range(len(templates)):
        # captions.append(templates[j].replace("{classname}", classes[i]))
    captions.append(templates[0][0] + classes[i] + templates[0][1])
text = tokenizer(captions).to(device=device)

with torch.no_grad(), torch.cuda.amp.autocast():
    correct_counter = 0
    total_counter = 0
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    for sample in ds:
        image = sample["image"]
        label = sample["label"]
        image = preprocess_val(image).unsqueeze(0).to(device=device)
        
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        max_prob, index = text_probs[0].max(dim=-1)
        # is_correct = classes[index]
        # remove punctuation and capitalize
        # label = label.replace("_", " ")
        new_label = label.replace("_", " ").capitalize()
        is_correct = classes[index].replace("_", " ").capitalize() == new_label
        # print(f"Predicted: {classes[index]}, Actual: {new_label}, Probability: {max_prob.item():.4f}")
        if is_correct:
            correct_counter += 1
        else:
            # print(f"Predicted: {classes[index]}, Actual: {label}")
            print(f"Incorrectly Predicted: {classes[index]}, Actual: {new_label}, Probability: {max_prob.item():.4f}")
        total_counter += 1

    print(f"Accuracy: {correct_counter / total_counter * 100:.2f}%")
    print(f"Total samples: {total_counter}, Correct predictions: {correct_counter}")

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

# top_five_probs, indices = torch.topk(text_probs[0], 5)
# top_five_labels = [classes[i] for i in indices]
# print("Top 5 probabilities:", top_five_probs)
# print("Top 5 labels:", top_five_labels)
# print("True label:", ds[0]["label"])

