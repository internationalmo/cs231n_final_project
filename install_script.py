import os
import subprocess
import sys

# Install requirements from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "--no-cache-dir", "install", "-r", "requirements.txt"])

# Import modules to ensure installation success
import torch
import torchvision
import datasets
import transformers
from PIL import Image
import accelerate
from datasets import load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image

print("Installation completed successfully.")