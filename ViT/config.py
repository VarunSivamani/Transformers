import os
import torch
from torch import nn
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_PATH = "./pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH + "/train"
TEST_DIR = IMAGE_PATH + "/test"
NUM_WORKERS = os.cpu_count()
IMG_SIZE = 224

MANUAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])   