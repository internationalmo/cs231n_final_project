import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoFeatureExtractor
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image
import accelerate
from datasets import load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


def get_split_dataset():
    dataset = load_dataset("blanchon/FireRisk")

    # Shuffle the data, then selct a subset
    dataset = dataset['train'].shuffle(seed=42).select(range(1000))

    # Split the dataset into training (80%), validation (10%), and test (10%)
    train_test_split = dataset.train_test_split(test_size=0.20)  # Splitting off 20% for validation + test
    test_val_split = train_test_split['test'].train_test_split(
        test_size=0.50)  # Split 50% of the 20% for test, which makes it 10% of the total

    # Creating a new DatasetDict to organize splits
    split_dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'],
        'test': test_val_split['test']
    })

    # Example of how to access these datasets
    print(f"Training Set Size: {len(split_dataset['train'])}")
    print(f"Validation Set Size: {len(split_dataset['validation'])}")
    print(f"Test Set Size: {len(split_dataset['test'])}")

    return split_dataset

def transform_dataset(split_dataset, selected_model, normalize=True, vertical_flip=False, horizontal_flip=False, vertical_p=0.5, horizontal_p=0.5, show_index=0):
    feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)
    
    transform_list = [Resize((224, 224)), ToTensor()]

    if vertical_flip:
        transform_list.append(RandomVerticalFlip(p=vertical_p))
    if horizontal_flip:
        transform_list.append(RandomHorizontalFlip(p=horizontal_p))
    if normalize:
        transform_list.append(Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std))

    _transforms = Compose(transform_list)
    
    def my_transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    fire_transformed = split_dataset.with_transform(my_transforms)

    # Show the before and after images for the given index
    original_image = split_dataset['train']['image'][show_index].convert("RGB")
    transformed_image = _transforms(original_image)
    transformed_image_pil = ToPILImage()(transformed_image)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(transformed_image_pil)
    ax[1].set_title('Transformed Image')
    ax[1].axis('off')

    plt.show()

    return fire_transformed, feature_extractor

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomVerticalFlip, RandomHorizontalFlip

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, RandomVerticalFlip, RandomHorizontalFlip

def show_image_transform_pipeline(split_dataset, selected_model, normalize=True, vertical_flip=1, horizontal_flip=1, index=0):
    feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)

    # Define individual transformations
    resize_transform = Resize((224, 224))
    to_tensor_transform = ToTensor()
    if normalize:
        normalize_transform = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    if vertical_flip:
        vertical_flip_transform = RandomVerticalFlip(p=1)
    if horizontal_flip:
        horizontal_flip_transform = RandomHorizontalFlip(p=1)
    
    # Load the original image
    original_image = split_dataset['train']['image'][index].convert("RGB")

    # Create a list to store images at each transformation step
    images = [original_image]
    titles = ['Original Image']

    # Apply each transformation step-by-step and store the results
    resized_image = resize_transform(original_image)
    images.append(resized_image)
    titles.append('Resized Image')

    tensor_image = to_tensor_transform(resized_image)
    
    if vertical_flip:
        flipped_image = vertical_flip_transform(tensor_image)
        images.append(ToPILImage()(flipped_image))
        titles.append('Vertical Flipped Image')
        tensor_image = flipped_image  # Update tensor_image for subsequent transformations

    if horizontal_flip:
        flipped_image = horizontal_flip_transform(tensor_image)
        images.append(ToPILImage()(flipped_image))
        titles.append('Horizontal Flipped Image')
        tensor_image = flipped_image  # Update tensor_image for subsequent transformations
    
    if normalize:
        normalized_image = normalize_transform(tensor_image)
        # Normalize transform keeps the tensor in the same format; displaying directly
        images.append(ToPILImage()(normalized_image))
        titles.append('Normalized Image')

    # Display images at each transformation step
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.show()


def train_model():
    available_models = {"resnet": "microsoft/resnet-50", "vit": "google/vit-base-patch16-224-in21k"}
    # Select which model
    selected_model = available_models["resnet"]
    split_dataset = get_split_dataset()
    #show_image_transform_pipeline(split_dataset, selected_model, normalize=True, vertical_flip=True, horizontal_flip=True, index=0)
    fire_transformed, feature_extractor = transform_dataset(split_dataset, selected_model,normalize=False, vertical_flip=False, horizontal_flip=True, vertical_p=1, horizontal_p=1, show_index=0)
    labels = split_dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

def main():
    train_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
