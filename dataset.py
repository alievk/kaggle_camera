import collections
from time import time
from pathlib import Path
from typing import Callable
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

import augmentations as aug
from augmentations import RandomManipulation


DATA_ROOT = Path('data')
SETS_ROOT = DATA_ROOT / 'sets'
TRAIN_SET = SETS_ROOT / 'train.csv'
VALID_SET = SETS_ROOT / 'valid.csv'
TRAINVAL_SET = SETS_ROOT / 'trainval.csv'
FLICKR_TRAIN_SET = SETS_ROOT / 'flickr_train.csv'
FLICKR_VALID_SET = SETS_ROOT / 'flickr_valid.csv'
REVIEWS_SET = SETS_ROOT / 'reviews.csv'
TRAINVAL_DIR = DATA_ROOT / 'train'
TEST_DIR = DATA_ROOT / 'test'
FLICKR_DIR = DATA_ROOT / 'external/flickr_images'
REVIEWS_DIR = DATA_ROOT / 'external/reviews_images'

CLASSES = ['HTC-1-M7', 'LG-Nexus-5x', 'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
           'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7', 'iPhone-4s', 'iPhone-6']
FLICKR_NAME_MAP = {'htc_m7': 'HTC-1-M7', 'nexus_5x': 'LG-Nexus-5x', 'moto_maxx': 'Motorola-Droid-Maxx',
                   'nexus_6': 'Motorola-Nexus-6', 'moto_x': 'Motorola-X', 'samsung_note3': 'Samsung-Galaxy-Note3',
                   'samsung_s4': 'Samsung-Galaxy-S4', 'sony_nex7': 'Sony-NEX-7', 'iphone_4s': 'iPhone-4s',
                   'iphone_6': 'iPhone-6'}
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = dict(zip(CLASSES, range(NUM_CLASSES)))
IDX_TO_CLASS = dict(zip(range(NUM_CLASSES), CLASSES))


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


class CSVDataset(Dataset):
    def __init__(self, csv_path, args, transform=None, class_aware=True, unique_samples=False,
                 do_manip=False, manip_prob: float=0.5, repeats=1, loader: Callable=opencv_loader,
                 stats_fq: int=0, fix_path: Callable=None):
        df = pd.read_csv(csv_path)
        paths = df['fname']
        is_external = 'flickr' in Path(csv_path).stem or 'reviews' in Path(csv_path).stem
        has_manip = 'manip' in df.columns
        samples = []
        class_samples = defaultdict(list)
        for i, path in enumerate(paths):
            if 'flickr' in Path(csv_path).stem:
                full_path = Path(FLICKR_DIR) / path
            elif 'reviews' in Path(csv_path).stem:
                full_path = Path(REVIEWS_DIR) / path
            else:
                full_path = Path(TRAINVAL_DIR / path)
            model = Path(path).parts[0]
            target = CLASS_TO_IDX[model] if not is_external else CLASS_TO_IDX[FLICKR_NAME_MAP[model]]
            item = {'path': str(full_path), 'target': target}
            if has_manip:
                item['manip'] = df.iloc[i]['manip']
            samples.append(item)
            class_samples[target].append(item)

        self.input_size = args.input_size
        self.transform = transform
        self.do_manip = do_manip
        self.manip_prob = manip_prob
        self.loader = loader
        self.samples = samples
        self.class_samples = class_samples
        self.repeats = repeats
        self.class_aware = class_aware
        self.unique_samples = unique_samples

        self.stats = collections.defaultdict(list)
        self.stats_fq = stats_fq
        self.stats_update_cnt = 0

        self.fix_path = fix_path

    def __getitem__(self, index):
        if self.class_aware:
            item = self._class_aware_sample(index)
        else:
            item = self.samples[index % len(self.samples)]
        path, target = item['path'], item['target']
        if self.fix_path:
            path = self.fix_path(path)

        load_s = time()
        img = self.loader(path)
        self.stats['loader'].append(time() - load_s)

        manip = -1
        if 'manip' in item and item['manip'] == 1:
            manip = aug.MANIPULATIONS.index('jpg70')
        elif self.do_manip and np.random.rand() < self.manip_prob:
            manip_s = time()
            disable_manip = []
            if img.shape[0] // 2 < self.input_size or img.shape[1] // 2 < self.input_size:
                disable_manip.append('bicubic0.5')
            img, manip_name = RandomManipulation()(img, disable_manip)
            manip = aug.MANIPULATIONS.index(manip_name)
            self.stats['manip'].append(time() - manip_s)

        if self.transform:
            tform_s = time()
            img = self.transform(img)
            self.stats['tform'].append(time() - tform_s)

        self.update_stats()

        return img, target, manip

    def __len__(self):
        if self.class_aware and not self.unique_samples:
            return self.repeats * len(CLASSES) * max([len(v) for v in self.class_samples.values()])
        return self.repeats * len(self.samples)

    def _class_aware_sample(self, g_idx):
        cls = g_idx % len(CLASSES)
        idx = g_idx // len(CLASSES)
        if idx and idx % len(self.class_samples[cls]) == 0:
            self.class_samples[cls] = np.random.permutation(self.class_samples[cls])
        return self.class_samples[cls][idx % len(self.class_samples[cls])]

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
    def __init__(self, image_dir=TEST_DIR, transform=None, loader: Callable=opencv_loader):
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
