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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "data"

DIM = (256, 256)
WIDTH, HEIGHT = DIM
NUM_CLASSES = 5
NUM_WORKERS = 24
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
SEED = 1

DEVICE = 'cuda'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.224]


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


def get_transforms(value='train'):
    if value == 'train':
        return albumentations.Compose([
            albumentations.Resize(HEIGHT, WIDTH),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=(-90, 90)),
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


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
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

    # use gpu if any
    net = net.cuda() if DEVICE else net
    return net


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loaders: dict,
            criterion,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.data_loaders = data_loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, inputs, labels, phase):
        # reset (or) zero the parameter gradients
        self.optimizer.zero_grad()
        # training (or) validation process
        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backpropagation in the network
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            return loss, preds

    def _run_epoch(self, epoch):
        #train_loader = self.data_loaders['train']
        #val_loader = self.data_loaders['val']
        batch_size = len(next(iter(self.data_loaders['train']))[0])
        steps = len(self.data_loaders['train'])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {steps}")
        #train_loader.sampler.set_epoch(epoch)
        #val_loader.sampler.set_epoch(epoch)
        # each epoch have training and validation phase
        for phase in ['train', 'val']:
            self.data_loaders[phase].sampler.set_epoch(epoch)
            # set mode for model
            if phase == 'train':
                self.model.train()  # set model to training mode
            else:
                self.model.eval()  # set model to evaluate mode

            running_loss = 0
            running_corrects = 0
            fin_out = []

            for inputs, labels in self.data_loaders[phase]:
                # move data to corresponding hardware
                inputs = inputs.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                loss, preds = self._run_batch(inputs, labels, phase)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # calculate loss and accuracy for the epoch
            epoch_loss = running_loss / len(self.data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(self.data_loaders[phase].dataset)

            # print loss and acc for training & validation
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        # set starting time
        start_time = time.time()

        for epoch in range(max_epochs):
            print(f'Epoch {epoch}/{max_epochs - 1}')
            print('-' * 15)
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        end_time = time.time() - start_time
        print('Training completes in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)

    train_dataset = CassavaDataSet(DATA_DIR, dimension=None, augmentations=None, dataset='train')
    train_loader = prepare_dataloader(train_dataset, batch_size)

    val_dataset = CassavaDataSet(DATA_DIR, dimension=None, augmentations=None, dataset='val')
    val_loader = prepare_dataloader(val_dataset, batch_size)

    data_loaders = {'train': train_loader, 'val': val_loader}

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    trainer = Trainer(model, data_loaders, criterion, optimizer, rank, save_every)
    trainer.train(total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
