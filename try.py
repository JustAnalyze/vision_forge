import time
from PIL import Image
from datetime import datetime
import json
import sys
from threading import Thread
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from icecream import ic

model_1 = torchvision.models.mobilenet_v3_large()
model_2 = torchvision.models.mobilenet_v3_large()

print(model_1.__class__.__name__)

ic(model_1.classifier, model_2.classifier[1],)