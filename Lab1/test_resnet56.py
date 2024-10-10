import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import resnet


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
model_name = 'resnet56'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_model(model_path):
    model = torch.nn.DataParallel(resnet.__dict__[model_name]())
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    return model


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave('test_image.png', np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted

if __name__ == "__main__":
    model_path = f'/content/drive/MyDrive/Lab1/save_resnet56/model.th'
    
    model = load_model(model_path)

    images, labels = next(iter(testloader)) 

    print("Showing random image from CIFAR-10 test dataset:")
    imshow(images[0])

    predicted_label = predict(model, images)
    print(f"Predicted: {classes[predicted_label.item()]}")
    print(f"Actual: {classes[labels[0]]}")
