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

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom function to convert the loaded dataset into a format usable by PyTorch
def transform_example(example):
    image = transform(example['image'])
    # print(type(image))
    label = example['label']
    return {'image': image, 'label': label}


# Applying transformation

def get_split_dataset():
    dataset = load_dataset("blanchon/FireRisk")

    # Split the dataset into training (80%), validation (10%), and test (10%)
    train_test_split = dataset['train'].train_test_split(test_size=0.20)  # Splitting off 20% for validation + test
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


def transform_dataset(split_dataset):
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

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


def train_model():
    split_dataset = get_split_dataset()
    fire_transformed, feature_extractor = transform_dataset(split_dataset)
    labels = split_dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=300,
        eval_steps=300,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
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

    trainer.train()


def main():
    train_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
