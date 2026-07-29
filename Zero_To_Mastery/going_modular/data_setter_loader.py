import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

def data_setter_loader (train_dir,
                        test_dir,
                        transformer,
                        BATCH_SIZE,
                        NUM_WORKERS=os.cpu_count(),
                        pin_memory=True,
                        ):
    """making a transform and imagefloder and Data loader and get classes"""
    #make the transform and the image folder
    train_data = ImageFolder(train_dir, transform=transformer)
    test_data = ImageFolder(test_dir, transform=transformer)
    #make the Data loader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=pin_memory)

    return train_loader, test_loader, train_data.classes






