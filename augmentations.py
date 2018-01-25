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
        return img[y1:y1 + th, x1:x1 + th, :].astype(np.uint8)


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
    def __call__(self, img):
        np.random.seed()  # this is important when multiprocessing
        manip = np.random.choice(MANIPULATIONS)  # type: str
        return manipulation(img, manip), manip
