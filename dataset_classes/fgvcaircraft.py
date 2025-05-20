import os
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class FGVCAircraftClass(Dataset):
    root = Path.home() / "tmp/Datasets/FGVCAircraft/fgvc-aircraft-2013b"
    base_folder = 'images'

    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        self._load_metadata()
        self.loader = default_loader

    def _load_metadata(self):
        # Load the appropriate CSV file based on the train flag
        if self.train:
            csv_file = os.path.join(self.root, 'train.csv')
        else:
            csv_file = os.path.join(self.root, 'test.csv')
            # Optionally include validation data in the test set
            val_csv_file = os.path.join(self.root, 'val.csv')
            if os.path.exists(val_csv_file):
                val_data = pd.read_csv(val_csv_file)
                test_data = pd.read_csv(csv_file)
                self.data = pd.concat([test_data, val_data], ignore_index=False)
                return
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample['filename'])
        target = sample['Labels']  # Labels are categorical values from 0 to 99
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target
