import torch
from torchvision import transforms
from datasets import Dataset, Image
import json
import numpy as np
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

def image_transform(x, transform_type):
        if transform_type == "grayscale":
            transformed_images = [transforms.Grayscale(num_output_channels=3)(image) for image in x["image"]]
        elif transform_type == "invert":
            transformed_images = [ImageOps.invert(image) for image in x["image"]]
        elif transform_type == "posterize":
            transformed_images = [ImageOps.posterize(image, bits=2) for image in x["image"]]
        else:
            transformed_images = x["image"]

        x["image"] = transformed_images
        return x

def preprocess_fn(x, preprocess):
    new_images = [preprocess(image) for image in x["image"]]
    x["image"] = new_images
    return x

def process_cifar100(dataset, json, preprocess, transform=None, semantic_shift='', semantic_shuffle=False, shift_shuffle=0):
    '''
    {
    'img': PIL.Image.Image,
    'fine_label': 0,
    'coarse_label': 5
    }'''
    classes = json["classes"]
    if semantic_shuffle:
        np.random.seed(1)
        np.random.shuffle(classes)
    classes_to_index = {classes[i]: i for i in range(len(classes))}
    index_to_classes = {i: classes[i] for i in range(len(classes))}

    if shift_shuffle > 0:
        new_front = classes[-shift_shuffle:]
        classes = classes[:-shift_shuffle]
        classes = new_front + classes

    assert len(classes) == 100, "CIFAR100 dataset should have 100 classes"

    # Prep the dataset columns
    dataset = dataset.cast_column("img", Image())
    dataset = dataset.rename_column("img", "image")
    dataset = dataset.rename_column("fine_label", "index_label")

    dataset.set_format(type="torch", columns=["image", "index_label"])

    transform_fn = transforms.Compose([
        transforms.Lambda(lambda x: image_transform(x, transform)),
        transforms.Lambda(lambda x: preprocess_fn(x, preprocess))
    ])
    dataset.set_transform(transform_fn)

    templates = json["templates"]
    captions = []
    for i in range(len(classes)):
        t = []
        for j in range(len(templates)):
            t.append(templates[j][0] + classes[i] + semantic_shift + templates[j][1])
        captions.append(t)
    
    return dataset, classes_to_index, index_to_classes, captions

def process_cifarc(dataset, json, preprocess, transform=None, semantic_shift='', semantic_shuffle=False, shift_shuffle=0):
    '''
    {
    'image': PIL.Image.Image,
    'label': 0,
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x7F8FAD2E3080>, 'label': 83, 'corruption_name': 'snow', 'corruption_level': 1}
    }'''
    classes = json["classes"]
    if semantic_shuffle:
        np.random.seed(1)
        np.random.shuffle(classes)
    classes_to_index = {classes[i]: i for i in range(len(classes))}
    index_to_classes = {i: classes[i] for i in range(len(classes))}

    if shift_shuffle > 0:
        new_front = classes[-shift_shuffle:]
        classes = classes[:-shift_shuffle]
        classes = new_front + classes

    assert len(classes) == 100, "CIFAR100 dataset should have 100 classes"

    # Prep the dataset columns
    dataset = dataset.cast_column("image", Image())
    # dataset = dataset.rename_column("img", "image")
    import ipdb; ipdb.set_trace()
    dataset = dataset.rename_column("label", "index_label")

    dataset.set_format(type="torch", columns=["image", "index_label"])

    transform_fn = transforms.Compose([
        transforms.Lambda(lambda x: image_transform(x, transform)),
        transforms.Lambda(lambda x: preprocess_fn(x, preprocess))
    ])
    dataset.set_transform(transform_fn)

    templates = json["templates"]
    captions = []
    for i in range(len(classes)):
        t = []
        for j in range(len(templates)):
            t.append(templates[j][0] + classes[i] + semantic_shift + templates[j][1])
        captions.append(t)
    
    return dataset, classes_to_index, index_to_classes, captions

def process_imagenet(dataset, json, preprocess, transform=None):
    '''
    ds['train'][0]
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=817x363 at 0x7F58FED23B90>, 'label': 726}
    '''
    classes = json["classes"]
    classes_to_index = {classes[i]: i for i in range(len(classes))}
    index_to_classes = {i: classes[i] for i in range(len(classes))}

    # Apply just-in-time transformation (no preprocessing, fast loading)
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.rename_column("label", "index_label")
    dataset.set_format(type="torch", columns=["image", "index_label"])

    transform_fn = transforms.Compose([
        transforms.Lambda(lambda x: image_transform(x, transform)),
        transforms.Lambda(lambda x: preprocess_fn(x, preprocess))
    ])
    dataset.set_transform(transform_fn)

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