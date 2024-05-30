import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoFeatureExtractor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer, TrainerCallback, AutoConfig
from PIL import Image
from copy import deepcopy
import accelerate
from datasets import load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def get_split_dataset(num_examples):
    dataset = load_dataset("blanchon/FireRisk")

    # Shuffle the data, then selct a subset
    dataset['train'] = dataset['train'].shuffle(seed=42)
    if num_examples is not None:
        dataset['train'] = dataset['train'].select(range(num_examples))

    # Split the dataset into training (80%), validation (10%), and test (10%)
    # train_test_split = dataset.train_test_split(test_size=0.20, stratify_by_column="label")  # Splitting off 20% for validation + test
    train_test_split = dataset['train'].train_test_split(test_size=0.20, stratify_by_column="label")  # Splitting off 20% for validation + test
    test_val_split = train_test_split['test'].train_test_split(
        test_size=0.50, stratify_by_column="label")  # Split 50% of the 20% for test, which makes it 10% of the total

    # Creating a new DatasetDict to organize splits
    split_dataset_train = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'],
        'test': test_val_split['test']
    })

    split_dataset_pretrain = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'],
        'test': test_val_split['test']
    })

    # Example of how to access these datasets
    print(f"Training Set Size: {len(split_dataset_train['train'])}")
    print(f"Validation Set Size: {len(split_dataset_train['validation'])}")
    print(f"Test Set Size: {len(split_dataset_train['test'])}")

    return split_dataset_train, split_dataset_pretrain
    # return split_dataset_train

def transform_dataset(split_dataset, selected_model, args, image_size):
    feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)
    
    transform_list = [Resize((image_size, image_size)), ToTensor()]

    if args.vertical_p > 0:
        transform_list.append(RandomVerticalFlip(p=args.vertical_p))
    if args.horizontal_p > 0:
        transform_list.append(RandomHorizontalFlip(p=args.horizontal_p))
    if args.to_normalize:
        transform_list.append(Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std))

    _transforms = Compose(transform_list)
    
    def my_transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        if "image" in examples:
            del examples["image"]
        return examples

    fire_transformed = split_dataset.with_transform(my_transforms)
    return fire_transformed, feature_extractor

def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    recall_metric = load_metric("recall")
    precision_metric = load_metric("precision")

    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')

    return {
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision
    }

# Function to compute confusion matrix
def compute_confusion_matrix(trainer, eval_dataset, id2label, file_path):
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=1)
    cm = confusion_matrix(labels, preds, labels = [i for i in range(len(id2label))])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[str(i)] for i in range(len(id2label))])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    directory = os.path.dirname(file_path)

    # Create directory if it doesn't exist
    if file_path is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save plot to the file path
        plt.savefig(file_path)

    plt.show()


def count_labels(dataset_split, split_name,labels):
    label_counts = {label: 0 for label in labels}
    for label in dataset_split['label']:
        label_counts[labels[label]] += 1
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(f'Data Distribution in {split_name} Set')
    plt.xticks(rotation=45)
    plt.show()