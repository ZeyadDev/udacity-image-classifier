import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image

import time

import seaborn as sns


def get_loaders_with_class(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = test_transforms

    # TODO: Load the datasets with ImageFolder
    train_datasets =  datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=64, shuffle=True)

    return train_loader, test_loader, validation_loader, train_datasets.class_to_idx


def construct_model(arch, device, hidden_units, learning_rate, model_state_dict=None, optimizer_state_dict=None):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
    elif arch == 'vgg16':
        vgg16 = models.vgg16(pretrained=True)
        in_features = 25088
    else:
        raise Exception("these are the supported architectures: densenet121, alexnet, vgg16")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(in_features, hidden_units)),
                        ('relu1', nn.ReLU()),
                        ('dropout1',nn.Dropout(p=0.2)),
                        ('fc2', nn.Linear(hidden_units, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                    ]))


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if not model_state_dict == None and not optimizer_state_dict == None:
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    model.to(device)
    return model, criterion, optimizer

def train(model, optimizer, criterion, train_loader, validation_loader, device, print_every,  epochs):

    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {validation_loss/len(validation_loader):.3f}.. "
                      f"validation accuracy: {validation_accuracy/len(validation_loader):.3f}")
                running_loss = 0
                model.train()


def test(model, criterion, test_loader, device):
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Accuracy: {:.3f}".format((test_accuracy/len(test_loader))*100) )


def save_model_checkpoint(model, class_to_idx, arch, hidden_units, learning_rate, optimizer, save_dir=None):
    model.class_to_idx = class_to_idx

    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'architecture': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate
    }

    file_path = 'checkpoint.pth'
    if not save_dir == None:
        file_path = save_dir+'/checkpoint.pth'
    torch.save(checkpoint, file_path)


def load_model_checkpoint(file_path, device):
    checkpoint = torch.load(file_path)

    class_to_idx = checkpoint['class_to_idx']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    arch = checkpoint['architecture']
    hidden_units =checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']

    model, _, __ = construct_model(arch, device, hidden_units, learning_rate, model_state_dict, optimizer_state_dict)
    
    return model, class_to_idx



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    size = 224
    width, height = image.size

    ratio = max(width/height, height/width)

    resize = [256, ratio*256]
    if width > height:
        resize = [ratio * 256, 256]

    image.thumbnail(size=resize)

    center = width/4, height/4
    image = image.crop((width/4-(244/2), height/4-(244/2), width/4+(244/2), height/4+(244/2)))


    image_arr = np.array(image)/255

    mean = [0.485, 0.456, 0.406]
    standard_dev = [0.229, 0.224, 0.225]

    np_image = (image_arr-mean) / standard_dev

    np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(model, image_path, device, class_to_idx, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file

    model.to('cpu')

    unpreprocessed_image = Image.open(image_path)

    image = process_image(unpreprocessed_image)

    img_torch = torch.from_numpy(image)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    img_torch.to('cpu')


    log_probs = model(img_torch)
    lin_probs = torch.exp(log_probs)

    top_p, top_index = lin_probs.topk(topk, dim=1)
    top_class = [class_to_idx[str(index)] for index in top_index[0].tolist()]
    return top_p, top_class
