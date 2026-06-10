from torchvision import transforms

# PathMNIST mean/std (precomputed from training split)
MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1270, 0.1542, 0.1196]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])