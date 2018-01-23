import time
import random
import shutil
import numpy as np
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as N
import torch.utils.data as D
from torch.autograd import Variable


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(init_optimizer, n_epochs=None, patience=2, lr_decay=0.2, max_lr_changes=2, **kwargs):
    args = kwargs['args']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    model = kwargs['model']
    criterion = kwargs['criterion']
    n_epochs = n_epochs or args.epochs

    output_dir = Path(args.output_dir)
    model_path = output_dir / 'model.pt'
    best_model_path = output_dir / 'best-model.pt'
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 0
        step = 0
        best_valid_loss = float('inf')

    save_checkpoint = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    lr = args.lr
    optimizer = init_optimizer(lr)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_avg = AverageMeter()
    log = output_dir.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    lr_changes = 0
    for epoch in range(epoch, n_epochs):
        print('Epoch {}/{}\t'
              'learning rate {}'.format(epoch + 1, n_epochs, lr))
        model.train()
        random.seed()
        try:
            end = time.time()
            for i, (input, target) in enumerate(train_loader):
                # measure data loading time
                if i > 0:  # first iteration is always slow
                    data_time.update(time.time() - end)

                input = Variable(input).cuda(async=True)
                target = Variable(target).cuda(async=True)

                output = model(input)  # type: Variable
                loss = criterion(output, target)

                optimizer.zero_grad()
                batch_size = input.size(0)
                (batch_size * loss).backward()
                optimizer.step()

                loss_avg.update(loss.data[0], input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    write_event(log, step, loss=loss_avg.avg)
                    print('\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           batch_time=batch_time, data_time=data_time, loss=loss_avg))

                step += 1

            write_event(log, step, loss=loss_avg.avg)
            save_checkpoint(epoch)

            valid_metrics = validation(valid_loader, model, criterion)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            print('Validation loss {m[valid_loss]}:.4f}\t'
                  'Score {m[score]:.4f}'.format(m=valid_loss))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                lr_changes += 1
                if lr_changes > max_lr_changes:
                    print('LR changes exceeded maximum, stop.')
                    break
                print('Validation loss plateaued, decaying LR.')
                lr *= lr_decay
                lr_reset_epoch = epoch
                optimizer = init_optimizer(lr)
                write_event(log, step, lr=lr)
        except KeyboardInterrupt:
            print('Ctrl+C, saving snapshot')
            save_checkpoint(epoch)
            break
    log.close()
    print('Done.')


def validation(loader: D.DataLoader, model: N.Module, criterion):
    # TODO: предусмотреть unmanip/manip примеры в скоре
    model.eval()
    losses = []
    targets = []
    outputs = []
    for input, target in loader:
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target).cuda()

        output = model(input_var)
        loss = criterion(output, target_var)
        losses.append(loss.data[0])

        output_data = output.data.cpu().numpy().argmax(axis=1)
        target_data = target.cpu().numpy()
        outputs.extend(list(output_data))
        targets.extend(list(target_data))

    loss = np.mean(losses)
    accuracy = np.mean(np.array(outputs) == np.array(targets))
    return {'valid_loss': loss, 'score': accuracy}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
