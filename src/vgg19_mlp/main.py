
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image

from os import listdir, walk
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import datasets
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter


# In[2]:


def mlp_module():
    mlp = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.1),
        nn.Linear(4096, 2)
    )
    return mlp


# In[3]:


def train(net, criterion, optimizer, traindir, devdir):
    # 4780
    # Mean for channels 0, 1, 2: 107.27337846
    # Std for channels 0, 1, 2: 78.4173511375
    mean = 116.987954934 / 256; std = 71.5262653842 / 256
    trainLoader = DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    # 199
    # Mean for channels 0, 1, 2: 130.181926013
    # Std for channels 0, 1, 2: 62.5028782841
    mean, std = 130.935603595 / 256, 60.4500025546 / 256
    devLoader = DataLoader(
        datasets.ImageFolder(devdir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    print("create DataLoader successfully!")
    
    net.train()
    wallclock = 1
    for ep in range(EPOCHS):
        running_loss = 0
        for i, data_train in enumerate(trainLoader):
            x, y = data_train
            x, y = Variable(x), Variable(y)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_p = net.forward(x)
            loss = criterion(y_p, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if (i + 1) % 10 == 0:
                running_loss /= 10
                print("epoch: {0}, step: {1}, training loss: {2}".format(ep + 1, i + 1, running_loss))
                writer.add_scalar('training loss', running_loss, wallclock)
                if (wallclock + 1) % 10 == 0:
                    running_loss, acc = validate(net, criterion, devLoader)
                    print("epoch: {0}, step: {1}, validation loss: {2}, validation acc: {3}".format(ep + 1, i + 1, running_loss, acc))
                    writer.add_scalar('validation loss', running_loss, wallclock)
                    writer.add_scalar('validation accuracy', acc, wallclock)
                wallclock += 1
                running_loss = 0
    torch.save(net.state_dict(), './vgg19Model')


# In[4]:


def validate(net, criterion, devLoader):
    net.eval()
    running_loss, bsz = 0, 0
    correct_cnt, total_cnt = 0, 0
    for data_dev in devLoader:
        x, y = data_dev
        x, y = Variable(x), Variable(y)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        y_p = net.forward(x)
        _, pred_label = torch.max(y_p.data, 1)
        total_cnt += x.size(0)
        correct_cnt += (pred_label == y.data).sum()
        loss = criterion(y_p, y)
        bsz += 1
        running_loss += loss.data[0]
    running_loss /= bsz
    acc = correct_cnt / total_cnt
    net.train()
    return running_loss, acc


# In[5]:


def test(net, featFile):
    net.eval()
    testSet = CustomDataset(featFile, None, isTrain=False)
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, collate_fn=collate_fn)
    predictions = []
    for i, data in enumerate(testLoader):
        x = data
        x = Variable(x)
        if torch.cuda.is_available():
            x = x.cuda()
        y_p = net.forward(x)
        _, pred_label = torch.max(y_p.data, 1)
        predictions.append(pred_label)
    return predictions


# In[ ]:


# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# mytransform = transforms.Compose([transforms.ToTensor()])
# cifarSet_train = dset.CIFAR10(root = "../../cifar/train/", train=True, download=True, transform=mytransform)
# cifarLoader_train = DataLoader(cifarSet_train, batch_size=10, shuffle=True, num_workers=2)

# cifarSet_dev = dset.CIFAR10(root = "../../cifar/dev/", train=False, download=True, transform=mytransform)
# cifarLoader_dev = DataLoader(cifarSet_dev, batch_size=10, shuffle=True, num_workers=2)
# for data_dev in cifarLoader_dev:
#     x, y = data_dev
#     print(x.size(), y.size())
#     break


# In[ ]:


if torch.cuda.is_available():
    print("using cuda")

finetune = True
BATCH_SIZE, EPOCHS, NUM_WORKERS = 32, 12, 1
net_vgg19 = models.vgg19(pretrained=finetune)
for param in net_vgg19.parameters():
    param.requires_grad = False
# Parameters of newly constructed modules have requires_grad=True by default
net_vgg19.classifier = mlp_module()

writer = SummaryWriter()

# use vgg19
net = net_vgg19
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    net, criterion = net.cuda(), criterion.cuda()
optimizer = optim.Adam(net.classifier.parameters(), lr=1e-3, weight_decay=1e-5)
train(net, criterion, optimizer, 'Training', 'TestImages')
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
print("training completed!")


# In[8]:


torch.save(net.state_dict(), "updateVGGmodel")

