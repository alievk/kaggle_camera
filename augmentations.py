import numbers

import numpy as np
import cv2


class CenterCrop:
    """Crops the given np.array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        # w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1 + th, x1:x1 + tw, :].astype(np.uint8)


class RandomCrop(object):
    """Crops the given np.array randomly to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        # w, h = img.size
        th, tw = self.size
        x1 = np.random.randint(0, w - tw - 1)
        y1 = np.random.randint(0, h - th - 1)
        return img[y1:y1 + th, x1:x1 + tw, :].astype(np.int64)


def rotate_90n_cw(src, angle):
    """Rotate image by angle which is multiple of 90"""
    assert angle % 90 == 0 and 360 >= angle >= -360
    if angle == 270 or angle == -90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    elif angle == 180 or angle == -180:
        dst = cv2.flip(src, -1)
    elif angle == 90 or angle == -270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)
    elif angle == 360 or angle == 0 or angle == -360:
        dst = np.copy(src)
    else:
        raise ValueError
    return dst


class RandomRotation:
    def __call__(self, img):
        np.random.seed()  # this is important when multiprocessing
        angles = [0, 90, 180, 270]
        random_angle = np.random.choice(angles)

        if random_angle == 0:
            return img

        return rotate_90n_cw(img, random_angle)


MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']


def manipulation(img, manip):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if manip.startswith('bicubic'):
        scale = float(manip[7:])
        img_manip = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif manip.startswith('gamma'):
        gamma = float(manip[5:])
        img_manip = np.uint8(cv2.pow(img / 255., gamma) * 255.)
    elif manip.startswith('jpg'):
        quality = int(manip[3:])
        _, buf = cv2.imencode('.jpeg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_manip = cv2.imdecode(buf, -1)
    else:
        raise ValueError
    return img_manip


class RandomManipulation:
    def __call__(self, img, disable_manip=[]):
        np.random.seed()  # this is important when multiprocessing
        manip_choices = [m for m in MANIPULATIONS if m not in disable_manip]
        manip = np.random.choice(manip_choices)  # type: str
        return manipulation(img, manip), manip
