# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
from torchvision import datasets, transforms, models
import torchvision.models as models
import json
import argparse 
    
#parser.add_argument('--input', action = 'store', default = 200)
#parser.add_argument('--num_categories', action= 'store', default =102)
#parser.add_argument('--dropout', dest ='dropout', action= 'store', default=0.5)
#parser.add_argument('--hidden_layer1', dest = 'hidden_layer1', action ='store',type=int, default=120)
#hidden1 = args.hidden_layer
#dropout = args.dropout
#num_categories = args.num_categories

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('--image_path', action= 'store', default = './flowers/test/1/image_06743.jpg')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--gpu', action= 'store_true', help="Use either CPU or CUDA")
parser.add_argument('--save_dir', dest ='save_dir', action= 'store', default='checkpoint.pth')
parser.add_argument('--learning_rate', dest ='learning_rate', action='store', default=0.0001)
parser.add_argument('--epochs', dest = 'epochs', action='store', type=int, default=5)
parser.add_argument('--arch', dest ='arch', action ='store', default ='alexnet', type=str) 
args = parser.parse_args()
image_path = args.image_path
data_dir = args.data_dir
path = args.save_dir
gpu_status = args.gpu
lr = args.learning_rate
arch= args.arch
epochs = args.epochs

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


def load_data(train_dir, valid_dir, test_dir):
        
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
        test_transforms =transforms.Compose([transforms.Resize(255),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
            
            
        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

        return train_data, test_data, test_data
        
train_data, valid_data, test_data = load_data(train_dir, valid_dir, test_dir)

def loader_function(train_data, valid_data, test_data):
   trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
   validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
   testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
        
   
   return trainloader, validloader, testloader
        #train_data, train_dir, valid_dir, test_dir, class_to_idx
trainloader, validloader, testloader = loader_function(train_data, valid_data, test_data)



   
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
 
structures = {"vgg16":25088,"densenet121" : 1024, "alexnet" : 9216 }

def nn_setup(structure='vgg16', dropout=0.05, hidden_layer1 = 500, lr = 0.0001):

        if structure == 'vgg16':
            model = models.vgg16(pretrained=True)        
        elif structure == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif structure == 'alexnet':
            model = models.alexnet(pretrained = True)
        else:
            print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))


        for param in model.parameters():
           param.requires_grad = False
           from collections import OrderedDict
           classifier = nn.Sequential(OrderedDict([
                    #('dropout',nn.Dropout(dropout)),
                    ('inputs', nn.Linear(structures[structure], hidden_layer1)),
                    ('relu1', nn.ReLU()),
                    ('dropout',nn.Dropout(dropout)),
                    ('hidden_layer1', nn.Linear(hidden_layer1, 256)),
                    ('relu2',nn.ReLU()),
                    #('hidden_layer2',nn.Linear(256,128)),
                    #('relu3',nn.ReLU()),
                    ('hidden_layer3',nn.Linear(256,102)),
                    ('output', nn.LogSoftmax(dim=1))
                                  ]))

           model.classifier = classifier
           criterion = nn.NLLLoss()
           optimizer = optim.Adam(model.classifier.parameters(), lr )
           model.cuda()

           return model , optimizer ,criterion 


model, optimizer, criterion = nn_setup('alexnet')
    
#old train:
epochs = 5
steps = 0
running_loss = 0
print_every = 40


if torch.device('cuda'):
    model.to('cuda')
else:
    model.to('cpu')

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        model.to('cuda')
        # Move input and label tensors to the default device
        inputs, labels= inputs.to('cuda'), labels.to('cuda')
       
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to('cuda:0') , labels.to('cuda:0')
                    model.to('cuda:0')
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

            
            
# TODO: Do validation on the test set
def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    model.to('cuda:0')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
check_accuracy_on_test(testloader)            
            
            
            
            
            
#saving our model:   
model.class_to_idx = train_data.class_to_idx
model.to('cpu')
        
checkpoint = {'structure': 'alexnet',
              'hidden_layer1': '120',
              'state_dict': model.state_dict(),
              'model.class_to_idx' : train_data.class_to_idx,
              'opt_state': optimizer.state_dict(),
              'epochs': epochs}
torch.save(checkpoint, path)   

def load_model(path):
     model.to('cpu')
     checkpoint = torch.load('checkpoint.pth', map_location=('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')) 
     model.structure = checkpoint['structure']
     hidden_layer1 = checkpoint['hidden_layer1']
     model,_,_ =nn_setup(structure, 0.5, hidden_layer1)
     model.class_to_idx = checkpoint['model.class_to_idx']
     model.load_state_dict(checkpoint['state_dict'])
        
     model = load_model('checkpoint.pth')
     model.eval()
     return model

    # print(model)