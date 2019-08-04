import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
from torchvision import datasets, transforms, models
import torchvision.models as models
import json
from PIL import Image
import argparse
from train import load_data, loader_function

parser = argparse.ArgumentParser(description='train.py')    
parser.add_argument('--image_path', action= 'store', default = './flowers/test/1/image_06743.jpg')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--gpu', dest='gpu', action= 'store_true', default=False)
parser.add_argument('--save_dir', dest ='save_dir', action= 'store', default='checkpoint.pth')
parser.add_argument('--topk', dest = 'topk', action= 'store', default = '5', type=int)
args = parser.parse_args() 
topk = args.topk
image_path = args.image_path
data_dir = args.data_dir
path = args.save_dir
gpu_status = args.gpu


model = torch.load('checkpoint.pth', map_location=('cuda' if  (torch.cuda.is_available()) else 'cpu'))

def process_image(image):
                   
    pil_im = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                   ])
    
    #applying transformation on pil image   
    pil_trans = transform(pil_im)
    
    return pil_trans

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
#imshow(process_image('flowers/test/1/image_06754.jpg'))

def predict(image_path, model, topk):
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    processed_image.shape
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(5)
    
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
        
    np_top_labs = top_labs[0].numpy()    
    
    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))
    
    top_flowers = []
    for lab in top_labels:
        top_flowers.append(cat_to_name[str(lab)])
    
    return top_probs, top_labels, top_flowers

top_probs, top_labels, top_flowers = predict(image_path, model, topk)

image_path = '/home/workspace/ImageClassifier/flowers/test/34/image_06961.jpg'
predict(image_path, model, topk=5)