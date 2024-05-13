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


# Applying transformation

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

def transform_dataset(split_dataset, selected_model):
    feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model)

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    # _transforms = Compose([RandomResizedCrop(feature_extractor.size), ToTensor(), normalize])
    _transforms = Compose([Resize((224, 224)), ToTensor(), normalize])

    def my_transforms(examples):
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
def compute_confusion_matrix(trainer, eval_dataset, id2label):
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=1)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[str(i)] for i in range(len(id2label))])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
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

def train_model():
    available_models = {"resnet": "microsoft/resnet-50", "vit": "google/vit-base-patch16-224-in21k"}
    # Select which model
    selected_model = available_models["resnet"]

    split_dataset = get_split_dataset()
    fire_transformed, feature_extractor = transform_dataset(split_dataset, selected_model)
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
        output_dir="./results",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=1,
        fp16=True,
        save_steps=300,
        eval_steps=300,
        learning_rate=2e-4,
        save_total_limit=5,
        load_best_model_at_end = True,
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

    trainer.train()
    compute_confusion_matrix(trainer, fire_transformed["test"],id2label)
    
    # trainer.evaluate(fire_transformed["test"])


def main():
    train_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
