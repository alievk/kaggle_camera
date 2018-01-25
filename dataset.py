import collections
from time import time
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import augmentations as aug
from augmentations import CenterCrop, RandomManipulation


DATA_ROOT = Path('data')
SETS_ROOT = DATA_ROOT / 'sets'
TRAIN_SET = SETS_ROOT / 'train.csv'
VALID_SET = SETS_ROOT / 'valid.csv'

NUM_CLASSES = 10
INPUT_SIZE = 112


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def opencv_loader(path):
    img = cv2.imread(path)
    assert img is not None, 'Image is not loaded'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


minimal_transform = transforms.Compose([
    CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_valid_transform = transforms.Compose([
    minimal_transform
])


class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=minimal_transform, do_manip=True, loader=opencv_loader, stats_fq=0,
                 fix_path=None):
        df = pd.read_csv(csv_path)
        classes = df.columns
        class_to_idx = dict(zip(classes, range(len(classes))))
        samples = []
        for cls in classes:
            paths = df[cls].tolist()
            samples.extend(zip(paths, [class_to_idx[cls]] * len(paths)))
        assert (len(samples) == df.shape[0] * df.shape[1])

        self.transform = transform
        self.do_manip = do_manip
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.stats = collections.defaultdict(list)
        self.stats_fq = stats_fq
        self.stats_update_cnt = 0

        self.fix_path = fix_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.fix_path:
            path = self.fix_path(path)

        load_s = time()
        img = self.loader(path)
        self.stats['loader'].append(time() - load_s)

        manip = None
        if self.do_manip:
            manip_s = time()
            img, manip = RandomManipulation()(img)
            self.stats['manip'].append(time() - manip_s)

        if self.transform:
            tform_s = time()
            img = self.transform(img)
            self.stats['tform'].append(time() - tform_s)

        self.update_stats()

        if manip:
            return img, target, aug.MANIPULATIONS.index(manip)
        return img, target

    def __len__(self):
        return len(self.samples)

    def update_stats(self):
        if self.stats_fq > 0:
            if self.stats_update_cnt and self.stats_update_cnt % self.stats_fq == 0:
                total = 0
                print('===Dataset performance===')
                for k, v in self.stats.items():
                    t = np.mean(v)
                    total += t
                    print('{}:\t{:.6f}'.format(k, t))
                print('total:\t{:.6f}'.format(total))
                self.stats = collections.defaultdict(list)
        else:
            self.stats = collections.defaultdict(list)

        self.stats_update_cnt += 1
