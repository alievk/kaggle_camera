import collections
from time import time
from pathlib import Path
from typing import Callable

import numpy as np
import cv2
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import augmentations as aug
from augmentations import CenterCrop, RandomManipulation, RandomRotation


DATA_ROOT = Path('data')
SETS_ROOT = DATA_ROOT / 'sets'
TRAIN_SET = SETS_ROOT / 'train.csv'
VALID_SET = SETS_ROOT / 'valid.csv'
TRAINVAL_SET = SETS_ROOT / 'trainval.csv'
FLICKR_TRAIN_SET = SETS_ROOT / 'flickr_train.csv'
FLICKR_VALID_SET = SETS_ROOT / 'flickr_valid.csv'
TRAINVAL_DIR = DATA_ROOT / 'train'
TEST_DIR = DATA_ROOT / 'test'
FLICKR_DIR = DATA_ROOT / 'external/flickr_images'

CLASSES = ['HTC-1-M7', 'LG-Nexus-5x', 'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
           'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7', 'iPhone-4s', 'iPhone-6']
FLICKR_NAME_MAP = {'htc_m7': 'HTC-1-M7', 'nexus_5x': 'LG-Nexus-5x', 'moto_maxx': 'Motorola-Droid-Maxx',
                   'nexus_6': 'Motorola-Nexus-6', 'moto_x': 'Motorola-X', 'samsung_note3': 'Samsung-Galaxy-Note3',
                   'samsung_s4': 'Samsung-Galaxy-S4', 'sony_nex7': 'Sony-NEX-7', 'iphone_4s': 'iPhone-4s',
                   'iphone_6': 'iPhone-6'}
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = dict(zip(CLASSES, range(NUM_CLASSES)))
IDX_TO_CLASS = dict(zip(range(NUM_CLASSES), CLASSES))

INPUT_SIZE = 112


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def opencv_loader(path):
    img = cv2.imread(path)
    assert img is not None, 'Image {} is not loaded'.format(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


minimal_transform = transforms.Compose([
    CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_valid_transform = transforms.Compose([
    RandomRotation(),
    minimal_transform
])

test_transform = transforms.Compose([
    CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=minimal_transform, do_manip=False, manip_prob: float=0.5,
                 loader: Callable=opencv_loader, stats_fq: int=0, fix_path: Callable=None):
        df = pd.read_csv(csv_path)
        paths = df['fname']
        is_flickr = 'flickr' in Path(csv_path).stem
        has_manip = 'manip' in df.columns
        samples = []
        for i, path in enumerate(paths):
            full_path = Path(TRAINVAL_DIR / path) if not is_flickr else Path(FLICKR_DIR) / path
            model = Path(path).parts[0]
            target = CLASS_TO_IDX[model] if not is_flickr else CLASS_TO_IDX[FLICKR_NAME_MAP[model]]
            item = {'path': str(full_path), 'target': target}
            if has_manip:
                item['manip'] = df.iloc[i]['manip']
            samples.append(item)

        self.transform = transform
        self.do_manip = do_manip
        self.manip_prob = manip_prob
        self.loader = loader
        self.samples = samples

        self.stats = collections.defaultdict(list)
        self.stats_fq = stats_fq
        self.stats_update_cnt = 0

        self.fix_path = fix_path

    def __getitem__(self, index):
        item = self.samples[index]
        path, target = item['path'], item['target']
        if self.fix_path:
            path = self.fix_path(path)

        load_s = time()
        img = self.loader(path)
        self.stats['loader'].append(time() - load_s)

        manip = -1
        if 'manip' in item and item['manip'] == 1:
            # actually, the jpg quality ranges between 80-93, but we don't care about it right now
            manip = aug.MANIPULATIONS.index('jpg90')
        elif self.do_manip and np.random.rand() < self.manip_prob:
            manip_s = time()
            img, manip_name = RandomManipulation()(img)
            self.stats['manip'].append(time() - manip_s)
            manip = aug.MANIPULATIONS.index(manip_name)

        if self.transform:
            tform_s = time()
            img = self.transform(img)
            self.stats['tform'].append(time() - tform_s)

        self.update_stats()

        return img, target, manip

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


class TestDataset(Dataset):
    def __init__(self, image_dir=TEST_DIR, transform=test_transform, loader: Callable=opencv_loader):
        self.images = [p for p in Path(image_dir).glob('*.tif')]
        self.image_dir = image_dir
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        path = str(self.images[index])
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)

        return img, path

    def __len__(self):
        return len(self.images)
