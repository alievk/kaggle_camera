import os
import argparse
import shutil
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import tqdm

import torch
import torch.nn as N
import torch.optim as O
import torch.utils.data as D
from torch.autograd import Variable
import torchvision.models as M

import dataset
import models
import utils


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

    run_dir = Path(args.run_dir)
    model_path = run_dir / 'model.pt'
    best_model_path = run_dir / 'best-model.pt'
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

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    log = run_dir.joinpath('train.log').open('at', encoding='utf8')
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
    model.eval()
    losses = []
    targets = []
    manips = []
    outputs = []
    for input, target, manip in loader:
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target).cuda()

        output = model(input_var)
        loss = criterion(output, target_var)
        losses.append(loss.data[0])

        output_data = output.data.cpu().numpy().argmax(axis=1)
        target_data = target.cpu().numpy()
        manip_data = manip.cpu().numpy()
        outputs.extend(list(output_data))
        targets.extend(list(target_data))
        manips.extend(list(manip_data))

    loss = np.mean(losses)
    unmanip_mask = np.array(manips) == -1
    matches = np.array(outputs) == np.array(targets)
    accuracy_unmanip = np.mean(matches[unmanip_mask])
    accuracy_manip = np.mean(matches[~unmanip_mask])
    accuracy_weighted = 0.7 * accuracy_unmanip + 0.3 * accuracy_manip
    print('Validation loss {:.4f}\t'
          'Accuracy unmanip {:.4f}\tmanip {:.4f}\tweighted: {:.4f}'.format(
        loss, accuracy_unmanip, accuracy_manip, accuracy_weighted))
    return {'valid_loss': loss, 'score': accuracy_weighted}


def inference(loader: D.DataLoader, model: N.Module):
    model.eval()
    preds = []
    img_paths = []
    for batch_input, batch_img_paths in loader:
        batch_input_var = Variable(batch_input, volatile=True).cuda()
        batch_pred = model(batch_input_var)
        preds.extend(list(batch_pred.data.cpu().numpy()))
        img_paths.extend(list(batch_img_paths))

    return preds, img_paths


def save_predictions(preds, paths, args):
    import pickle
    import pandas as pd
    preds_cls, names = [], []
    for pred_prob, path in zip(preds, paths):
        pred_idx = np.argmax(pred_prob)
        pred_cls = dataset.IDX_TO_CLASS[pred_idx]
        name = os.path.basename(path)
        preds_cls.append(pred_cls)
        names.append(name)

    run_dir = Path(args.run_dir)
    csv_path = run_dir / (args.mode + '.csv')
    infer_path = run_dir / (args.mode + '_detailed.pkl')
    pd.DataFrame({'fname': names, 'camera': preds_cls}, columns=['fname', 'camera']).to_csv(str(csv_path), index=False)
    pickle.dump((preds, paths), open(str(infer_path), 'wb'))


def add_arguments(parser: argparse.ArgumentParser):
    arg = parser.add_argument
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'], default='train')
    arg('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    arg('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    arg('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency')
    arg('--lr', '--learning-rate', default=0.0002, type=float, metavar='LR', help='initial learning rate')
    arg('-r', '--run-dir', required=True, metavar='DIR', help='directory with model checkpoints, logs, etc.')
    arg('--clean', action='store_true', help='clean the output directory')
    arg('--patience', default=4, type=int, metavar='N',
        help='number of epochs without validation loss improvement to tolerate')
    arg('--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    arg('--checkpoint', choices=['best', 'last'], default='last',
        help='whether to use the best or the last model checkpoint for inference')


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    assert torch.cuda.is_available(), 'CUDA is not available'

    train_dataset = dataset.CSVDataset(dataset.TRAIN_SET, transform=dataset.train_valid_transform,
                                       do_manip=False, fix_path=utils.fix_jpg_tif)
    valid_dataset = dataset.CSVDataset(dataset.VALID_SET, transform=dataset.train_valid_transform,
                                       do_manip=True, fix_path=utils.fix_jpg_tif)

    train_loader = D.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    valid_loader = D.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    model = models.ResNet(num_classes=dataset.NUM_CLASSES, model_creator=M.resnet50, pretrained=True)
    model.cuda()

    loss = N.CrossEntropyLoss()

    if 'train' == args.mode:
        if run_dir.exists() and args.clean:
            shutil.rmtree(str(run_dir))
        run_dir.mkdir(exist_ok=True)
        run_dir.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        train_kwargs = {
            'args': args,
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'model': model,
            'criterion': loss,
        }
        train(
            init_optimizer=lambda lr: O.SGD(model.fresh_parameters(), lr=lr, momentum=0.9),
            n_epochs=1,
            **train_kwargs)
        train(
            init_optimizer=lambda lr: O.SGD(model.parameters(), lr=lr, momentum=0.9),
            **train_kwargs)
    elif args.mode in ['valid', 'predict_test']:
        if 'best' == args.checkpoint:
            ckpt = (run_dir / 'best-model.pt')
        else:
            ckpt = (run_dir / 'model.pt')
        state = torch.load(str(ckpt))
        model.load_state_dict(state['model'])
        print('Loaded {}'.format(ckpt))
        if 'valid' == args.mode:
            validation(tqdm.tqdm(valid_loader, desc='Validation'), model, loss)
        elif 'predict_test' == args.mode:
            test_dataset = dataset.TestDataset()
            test_loader = D.DataLoader(test_dataset, batch_size=32, num_workers=0)
            preds, paths = inference(test_loader, model)
            save_predictions(preds, paths, args)
    else:
        raise ValueError('Unknown mode {}'.format(args.mode))

if '__main__' == __name__:
    main()
