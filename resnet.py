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
from model_utils import *


def train_model(args):
    selected_model = "microsoft/resnet-50"
    split_dataset_train, split_dataset_pretrain  = get_split_dataset(args.num_examples)

    data_collator = DefaultDataCollator()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    if args.pretrain_model_dir is not None:
        print("In Pretraining")
        selected_model = "microsoft/resnet-50"
        fire_transformed_small, feature_extractor_small = transform_dataset(split_dataset_pretrain, selected_model, args, 32)
        labels = split_dataset_pretrain["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label


        model_pretrain = AutoModelForImageClassification.from_pretrained(selected_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes = True,
        )

        model_pretrain.to(device)

        training_args_pretrain = TrainingArguments(
            output_dir=args.pretrain_model_dir,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.pretrain_epochs,
            learning_rate=2e-4,
            remove_unused_columns=False,
        )

        trainer_pretrain = Trainer(
            model=model_pretrain,
            args=training_args_pretrain,
            data_collator=data_collator,
            train_dataset=fire_transformed_small["train"],
            eval_dataset=fire_transformed_small["validation"],
            compute_metrics=compute_metrics,
            tokenizer=feature_extractor_small,
        )
    # Plot label distribution
    #count_labels(split_dataset['train'], "Training",labels)
    #count_labels(split_dataset['validation'], "Validation",labels)
    #count_labels(split_dataset['test'], "Test",labels)

        trainer_pretrain.add_callback(CustomCallback(trainer_pretrain)) 
        trainer_pretrain.train()
        model_pretrain.save_pretrained(args.pretrain_model_dir)
    
    finetune_model = selected_model if args.pretrain_model_dir is None else args.pretrain_model_dir
    print(finetune_model)
    print("Finetuning")

    data_collator = DefaultDataCollator()

    fire_transformed_large, feature_extractor_large = transform_dataset(split_dataset_train, selected_model, args, 224)
    labels = split_dataset_train["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model_large = AutoModelForImageClassification.from_pretrained(finetune_model,
                                                                num_labels=len(labels),
                                                                id2label=id2label,
                                                                label2id=label2id,
                                                                ignore_mismatched_sizes=True)

    model_large.to(device)
    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.train_epochs,
        learning_rate=2e-4,
        remove_unused_columns=False,
    )

    trainer_large = Trainer(
        model=model_large,
        args=training_args,
        data_collator=data_collator,
        train_dataset=fire_transformed_large["train"],
        eval_dataset=fire_transformed_large["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor_large,
    )

    trainer_large.add_callback(CustomCallback(trainer_large))
    trainer_large.train()

    compute_confusion_matrix(trainer_large, fire_transformed_large["test"],id2label, args.confusion_matrix_path)
    
    # trainer.evaluate(fire_transformed["test"])


def main():

#   Example command to run with pretraining:
#        python3 resnet.py --to_normalize=1 --horizontal_p=0.5 --vertical_p=0  --model_output_dir="./dummy_results" --pretrain_model_dir="./dummy_results/pretrain" --confusion_matrix_path="./heatmaps/heatmap_v2.png" --num_examples=2000
#   Example command to run without pretraining:
#        python3 resnet.py --to_normalize=1 --horizontal_p=0.5 --vertical_p=0  --model_output_dir="./dummy_results"  --confusion_matrix_path="./heatmaps/heatmap_v2.png" --num_examples=2000 --pretrain_epochs=2 --train_epochs=2
    parser = argparse.ArgumentParser(description="Process file with different methods.")
    parser.add_argument("--to_normalize", type=int, choices=[1, 0], default=1, help="1 to normalize or 0 to not normalize. Default is to normalize")
    parser.add_argument("--horizontal_p", type=float, default=0.0, help="Probability of horzontal flipping. Default is to .5")
    parser.add_argument("--vertical_p", type=float, default=0.0, help="Probability of vertical flipping. Default is to .5")
    parser.add_argument("--model_output_dir", type=str, required=True, help="The directory path for the model results.")
    parser.add_argument("--confusion_matrix_path", type=str, default=None, help="The full path to save the confusion matrix. Defaults to 'None' so nothing is saved.")
    parser.add_argument("--num_examples", type=int, default=None, help="The total number of examples to use. Helpful for doing light testing.")
    parser.add_argument("--pretrain_model_dir", type=str, default=None, help="Location to save pretrained model.")
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    args = parser.parse_args()

    train_model(args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
