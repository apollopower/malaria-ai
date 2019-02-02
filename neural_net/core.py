import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
import requests
import urllib

# File to contain data science stuff 

def conv_model(model_path):
    conv_model = models.resnet50(pretrained=True)
    
    for param in conv_model.parameters():
        param.require_grad = False

    conv_model.fc = nn.Linear(2048, 2, bias=True)

    fc_parameters = conv_model.fc.parameters()

    for param in fc_parameters:
        param.requires_grad = True

    conv_model.load_state_dict(torch.load(model_path))
    return conv_model


def load_image(img_path):
    response = urllib.request.urlopen(img_path)
    image = Image.open(response)
    prediction_transform = transforms.Compose([transforms.Resize(size=(224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    # Discard transparent channel, and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image


def predict_malaria(model, class_names, img_path):
    # Load the image and return the predicted diagnosis
    img = load_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]
