import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class PathMNISTDataset(Dataset):
    def __init__(self, npz_path: str, split: str, transform=None):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]   # (N, 28, 28, 3) uint8
        self.labels = data[f"{split}_labels"].squeeze()  # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])  # PIL RGB
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label