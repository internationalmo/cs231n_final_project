import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoFeatureExtractor
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, TrainerCallback
from PIL import Image
from copy import deepcopy
import accelerate
from datasets import load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
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

def transform_dataset(split_dataset, selected_model, to_normalize):
    feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    if to_normalize:
        _transforms = Compose([Resize((224, 224)), ToTensor(), normalize])
    else:
        _transforms = Compose([Resize((224, 224)), ToTensor()])

    def my_transforms(examples):
        if "image" in examples:
            examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
            del examples["image"]
        return examples

    fire_transformed = split_dataset.with_transform(my_transforms)
    return fire_transformed, feature_extractor


def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

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

def train_model(args):
    available_models = {"resnet": "microsoft/resnet-50", "vit": "google/vit-base-patch16-224-in21k"}
    # Select which model
    selected_model = available_models[args.model]

    split_dataset = get_split_dataset(args.num_examples)
    fire_transformed, feature_extractor = transform_dataset(split_dataset, selected_model, args.to_normalize)
    labels = split_dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(selected_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes = True,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        num_train_epochs=5,
        fp16=True,
        learning_rate=2e-4,
        save_total_limit=5,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=fire_transformed["train"],
        eval_dataset=fire_transformed["test"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    # Plot label distribution
    count_labels(split_dataset['train'], "Training",labels)
    count_labels(split_dataset['validation'], "Validation",labels)
    count_labels(split_dataset['test'], "Test",labels)

    trainer.add_callback(CustomCallback(trainer)) 
    trainer.train()
    
    compute_confusion_matrix(trainer, fire_transformed["test"],id2label, args.confusion_matrix_path)
    
    # trainer.evaluate(fire_transformed["test"])


def main():

#   Example command to run:
#        python3 python3 baseline.py --model="resnet" --model_output_dir="./dummy_results" --confusion_matrix_path="./heatmaps/heatmap_v2.png"
    parser = argparse.ArgumentParser(description="Process file with different methods.")
    parser.add_argument("--model", type=str, choices=["resnet", "vit"], required=True, help="Choose either 'resnet' or 'vit'.")
    parser.add_argument("--to_normalize", type=int, choices=[1, 0], default=1, help="1 to normalize or 0 to not normalize. Default is to normalize")
    parser.add_argument("--model_output_dir", type=str, required=True, help="The directory path for the model results.")
    parser.add_argument("--confusion_matrix_path", type=str, default=None, help="The full path to save the confusion matrix. Defaults to 'None' so nothing is saved.")
    parser.add_argument("--num_examples", type=int, default=None, help="The total number of examples to use. Helpful for doing light testing.")
    args = parser.parse_args()

    train_model(args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
