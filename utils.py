class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, tau=0.5):
        self.tau = tau
        self.reset()

    def reset(self):
        self.avg = 0.
        self.val = 0.
        self.first = True

    def update(self, x):
        self.val = x
        if self.first:
            self.first = False
            self.avg = x
        else:
            self.avg = (1. - self.tau) * x + self.tau * self.avg


def fix_jpg_tif(p):
    if '.JPG' in p:
        p = p.replace('.JPG', '.tif')
    elif '.jpg' in p:
        p = p.replace('.jpg', '.tif')
    else:
        raise ValueError
    return p
