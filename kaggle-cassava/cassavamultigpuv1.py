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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


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


def train_model(gpu_id, model, dataloaders, criterion, optimizer, save_every, num_epochs=5):
    model = model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    # set starting time
    start_time = time.time()

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 15)
        batch_size = len(next(iter(dataloaders['train']))[0])
        steps = len(dataloaders['train'])
        print(f"[GPU{gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {steps}")

        # each epoch have training and validation phase
        for phase in ['train', 'val']:
            dataloaders[phase].sampler.set_epoch(epoch)
            # set mode for model
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                # move data to corresponding hardware
                inputs = inputs.to(gpu_id)
                labels = labels.to(gpu_id)

                # reset (or) zero the parameter gradients
                optimizer.zero_grad()

                # training (or) validation process
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backpropagation in the network
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

        if gpu_id == 0 and epoch % save_every == 0:
            _save_checkpoint(model, epoch)

    end_time = time.time() - start_time

    print('Training completes in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


def _save_checkpoint(model, epoch):
    ckp = model.module.state_dict()
    PATH = "checkpoint.pt"
    torch.save(ckp, PATH)
    print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


def main(rank: int, world_size: int, save_every: int, total_epochs: int):
    ddp_setup(rank, world_size)

    model = get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train_dataset = CassavaDataSet(
        data_dir=BASE_DIR,
        augmentations=get_transforms('train'),
        dimension=DIM
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_dataset)
    )

    val_dataset = CassavaDataSet(
        data_dir=BASE_DIR,
        augmentations=get_transforms('val'),
        dimension=DIM
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(val_dataset)
    )

    loaders = {'train': train_loader, 'val': val_loader}

    # train the model
    train_model(gpu_id=rank, model=model, dataloaders=loaders, criterion=criterion, optimizer=optimizer, save_every=save_every, num_epochs=total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, 1, 4), nprocs=world_size)
