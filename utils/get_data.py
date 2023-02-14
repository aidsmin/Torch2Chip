"""
dataset and data loader
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets

def get_loader(args):
    # Preparing data
    if args.dataset == 'cifar10':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        train_xforms = [transforms.RandomHorizontalFlip(), 
                        transforms.RandomCrop(32, padding=4)]

        train_xforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        train_transform = transforms.Compose(train_xforms)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes, img_size = 10, 32
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')

        train_data = datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = datasets.ImageFolder(test_dir, transform=test_transform)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        num_classes, img_size = 1000, 224
    else:
        raise ValueError("Unrecegonized dataset!")
    
    return trainloader, testloader, num_classes, img_size