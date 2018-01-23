from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


DATA_ROOT = Path('data')
SETS_ROOT = DATA_ROOT / 'sets'
TRAIN_SET = SETS_ROOT / 'train.csv'
VALID_SET = SETS_ROOT / 'valid.csv'

NUM_CLASSES = 10
INPUT_SIZE = 112


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.CenterCrop(INPUT_SIZE),
    img_transform
])

valid_transform = transforms.Compose([
    transforms.CenterCrop(INPUT_SIZE),
    img_transform
])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=img_transform, loader=pil_loader):
        df = pd.read_csv(csv_path)
        classes = df.columns
        class_to_idx = dict(zip(classes, range(len(classes))))
        samples = []
        for cls in classes:
            paths = df[cls].tolist()
            samples.extend(zip(paths, [class_to_idx[cls]] * len(paths)))
        assert (len(samples) == df.shape[0] * df.shape[1])

        self.transform = transform
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)
