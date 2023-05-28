import os
import copy
import json
import time
import pandas as pd
import numpy as np

# visualization modules
from PIL import Image

# pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

# augmentations
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import warnings

warnings.filterwarnings('ignore')

BASE_DIR = "data/"

DIM = (256, 256)
WIDTH, HEIGHT = DIM
NUM_CLASSES = 5
NUM_WORKERS = 24
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
SEED = 1

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.224]

def get_transforms(value ='val'):
    if value == 'train':
        return albumentations.Compose([
            albumentations.Resize(HEIGHT, WIDTH),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=(-90,90)),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
            ToTensorV2(p=1.0)
        ])
    elif value == 'val':
        return albumentations.Compose([
            albumentations.Resize(HEIGHT, WIDTH),
            albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
            ToTensorV2(p=1.0)
        ])


class CassavaDataSet(Dataset):

    def __init__(self, data_dir, dimension=None, augmentations=None, dataset: str = 'train'):
        super().__init__()
        self.data_dir = data_dir
        self.dimension = dimension
        self.augmentations = augmentations
        self.dataset = dataset

        if dataset == 'train':
            self.data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        else:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'val.csv'))

        self.images_dir = os.path.join(self.data_dir, "train_images")
        self.image_ids = list(self.data['image_id'])
        self.labels = list(self.data['label'])

    # returns the length
    def __len__(self):
        return len(self.image_ids)

    # return the image and label for the index
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.image_ids[idx]))

        if self.dimension:
            img = img.resize(self.dimension)

        # convert to numpy array
        img = np.array(img)

        if self.augmentations:
            augmented = self.augmentations(image=img)
            img = augmented['image']

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


train_dataset = CassavaDataSet(
    data_dir=BASE_DIR,
    augmentations=get_transforms('train'),
    dimension=DIM
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)

val_dataset = CassavaDataSet(
    data_dir=BASE_DIR,
    augmentations=get_transforms('val'),
    dimension=DIM
)

val_loader = DataLoader(
    val_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)


def get_model():
    net = models.resnet152(pretrained=True)

    # if you want to train the whole network, comment this code
    # freeze all the layers in the network
    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = net.fc.in_features
    # create last few layers
    net.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
        nn.LogSoftmax(dim=1)
    )

    return net


def train_model(device, model, dataloaders, criterion, optimizer, num_epochs=5):
    model = model.to(device)
    # set starting time
    start_time = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 15)

        # each epoch have training and validation phase
        for phase in ['train', 'val']:
            # set mode for model
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0
            running_corrects = 0
            fin_out = []

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                # move data to corresponding hardware
                inputs = inputs.to(device)
                labels = labels.to(device)

                # reset (or) zero the parameter gradients
                optimizer.zero_grad()

                # training (or) validation process
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backpropagation in the network
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # calculate loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print loss and acc for training & validation
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # update the best weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    end_time = time.time() - start_time

    print('Training completes in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
    print('Best Val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main(device=0):
    model = get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loaders = {'train': train_loader, 'val': val_loader}

    # train the model
    model, accuracy = train_model(device=device, model=model, dataloaders=loaders, criterion=criterion, optimizer=optimizer, num_epochs=4)

    # save the model and model weights
    torch.save(model, './best_model.h5')
    torch.save(model.state_dict(), './best_model_weights')


if __name__ == "__main__":
    device = 0
    main(device)
