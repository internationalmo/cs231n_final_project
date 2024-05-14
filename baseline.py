import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
from copy import deepcopy
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoFeatureExtractor
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, TrainerCallback
from PIL import Image
import accelerate
from datasets import load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # MAX: Start of new stuff.
        if "labels" in inputs:
            preds = outputs.logits.detach()
            acc = (
                (preds.argmax(axis=1) == inputs["labels"])
                .type(torch.float)
                .mean()
                .item()
            )
            self.log({"accuracy": acc})
        # MAX: End of new stuff.

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


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
    train_test_split = dataset['train'].train_test_split(test_size=0.20, stratify_by_column="label")  # Splitting off 20% for validation + test
    test_val_split = train_test_split['test'].train_test_split(
        test_size=0.50, stratify_by_column="label")  # Split 50% of the 20% for test, which makes it 10% of the total

    limited_train_examples = train_test_split['train'].train_test_split(test_size=0.6, stratify_by_column="label") 
    # Creating a new DatasetDict to organize splits
    # split_dataset = DatasetDict({
    #     'train': train_test_split['train'],
    #     'validation': test_val_split['train'],
    #     'test': test_val_split['test']
    # })
    split_dataset = DatasetDict({
        'train': limited_train_examples['train'],
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


def create_heatmap(predictions, true_labels, label_mapping, save_path):
    # Initialize a 2D array to count misclassifications
    data = np.zeros((len(label_mapping), len(label_mapping)), dtype=np.int32)

    # Iterate through predictions and true labels to count misclassifications
    for i in range(len(predictions)):
        pred_label = predictions[i]
        true_label = true_labels[i]
        data[true_label, pred_label] += 1


    plt.figure(figsize=(18, 16))
    sns.heatmap(data, fmt=".2f", cmap="YlGnBu", annot=True)

    # Set labels for x and y axes using label mapping
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.xticks(np.arange(0.5, len(data) + 0.5), [label_mapping[i] for i in range(len(data))], rotation=45)
    plt.yticks(np.arange(0.5, len(data) + 0.5), [label_mapping[i] for i in range(len(data))], rotation=0)

    # Save the plot if save_path is provided
    plt.title("Predicted Labels vs True Labels Heatmap")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def train_model():
    split_dataset = get_split_dataset()
    fire_transformed, feature_extractor = transform_dataset(split_dataset)

    # # Get the first 1000 examples
    # subset_train_dataset = fire_transformed["train"].select(range(200))

    # # Update the DatasetDict with the subset training dataset
    # fire_transformed["train"] = subset_train_dataset
    # fire_transformed["validation"] = fire_transformed["validation"].select(range(200))


    true_labels = np.array(fire_transformed["validation"]["label"])

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results/baseline_vit_22k_examples/",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        num_train_epochs=5,
        dataloader_num_workers = 1,
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
        eval_dataset=fire_transformed["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    trainer.add_callback(CustomCallback(trainer)) 
    trainer.train()
    predictions = trainer.predict(fire_transformed["validation"]).predictions
    
    # Get the true labels from the test dataset

    label_mapping = {
        0: "Very Low",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High",
        5: "Non-burnable",
        6: "Water"
    }

    create_heatmap(predictions, true_labels, label_mapping, "./results/baseline_vit_22k_examples/heatmaps/baseline_vit_heatmap.png")


def main():
    train_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
