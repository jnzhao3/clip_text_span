import torch
from datasets import Dataset, Image
import json
from PIL import ImageOps
##==== END OF IMPORTS ====##

##==== README ====##
'''
Data is processed to be of this form:
image = sample["image"]
label = sample["label"]
index_label = sample["index_label"]

'''
##==== END OF README ====##

def one_hot(n, i):
    '''
    Create torch tensor of size n with 1 at index i and 0 elsewhere.
    '''
    t = torch.zeros(n)
    t[i] = 1
    return t

def process_imagenet(dataset, json, transform=None):
    '''
    ds['train'][0]
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=817x363 at 0x7F58FED23B90>, 'label': 726}
    '''
    classes = json["classes"]
    classes_to_index = {classes[i]: i for i in range(len(classes))}
    index_to_classes = {i: classes[i] for i in range(len(classes))}

    dataset = dataset.cast_column("image", Image())
    # dataset = dataset.map(lambda x: {"index_label": x["label"]})
    dataset = dataset.rename_column("label", "index_label")
    if transform == "grayscale":
        image_map = lambda x: x.convert("L")
    elif transform == "invert":
        image_map = lambda x: ImageOps.invert(x)
    elif transform == "posterize":
        image_map = lambda x: ImageOps.posterize(x, bits=2)
    else:
        image_map = lambda x: {"image": x["image"]}
    dataset = dataset.map(lambda x: {"label": index_to_classes[x["index_label"]], "image": image_map(x["image"])})
    # if grayscale:
    #     dataset = dataset.map(lambda x: {"image": x["image"].convert("L")})

    templates = json["templates"]
    captions = []
    for i in range(len(classes)):
        t = []
        for j in range(len(templates)):
            t.append(templates[j][0] + classes[i] + templates[j][1])
        captions.append(t)
    
    return dataset, classes_to_index, index_to_classes, captions

def process_birds(dataset, json):
    '''
    Process the birdsnap dataset to extract images and labels.
    '''
    dataset = dataset.cast_column("image", Image())
    # Remove all underscores from the labels of the dataset
    dataset = dataset.map(lambda x: {"label": x["label"].replace("_", " ")})
    json_contents = json
    classes = json_contents["classes"]
    classes_to_index = {classes[i]: i for i in range(len(classes))}
    index_to_classes = {i: classes[i] for i in range(len(classes))}
    captions = []
    templates = json_contents["templates"]

    n = len(classes)
    dataset = dataset.map(lambda x: {"one_hot_label": one_hot(n, classes_to_index[x["label"]]), "index_label": classes_to_index[x["label"]]})
    for i in range(len(classes)):
        t = []
        for j in range(len(templates)):
            t.append(templates[j][0] + classes[i] + templates[j][1])
        captions.append(t)

    return dataset, classes_to_index, index_to_classes, captions