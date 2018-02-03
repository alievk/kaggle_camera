import os
import argparse
import shutil
import json
import time
import random
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import tqdm
import pickle
import pandas as pd

import torch
import torch.nn as N
import torch.optim as O
import torch.utils.data as D
from torch.autograd import Variable
import torchvision.transforms as transforms

import dataset
import models
import utils
from augmentations import CenterCrop, RandomRotation


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(init_optimizer, lr, n_epochs=None, lr_decay=0.2, max_lr_changes=2, **kwargs):
    args = kwargs['args']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    model = kwargs['model']
    criterion = kwargs['criterion']
    n_epochs = n_epochs or args.epochs
    patience = args.patience

    run_dir = Path(args.run_dir)
    model_path = run_dir / 'model.pt'
    best_model_path = run_dir / 'best-model.pt'
    if args.snapshot:
        snapshot = Path(args.snapshot)
    else:
        snapshot = model_path
    if snapshot.exists():
        state = torch.load(str(snapshot))
        epoch = state['epoch'] + 1
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        best_score = state['best_score']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 0
        step = 0
        best_valid_loss = float('inf')
        best_score = 0.

    save_checkpoint = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_score': best_score
    }, str(model_path))

    def save_best_checkpoint(metrics):
        print('Saving best checkpoint with loss {}, score {}'.format(
            metrics['score'], metrics['score']))
        shutil.copy(str(model_path), str(best_model_path))

    optimizer = init_optimizer(lr)

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    log = run_dir.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    lr_changes = 0
    write_event(log, step, lr=lr)
    for epoch in range(epoch, n_epochs):
        print('Epoch {}/{}\t'
              'learning rate {}'.format(epoch + 1, n_epochs, lr))
        model.train()
        random.seed()
        try:
            end = time.time()
            for i, (input, target, manip) in enumerate(train_loader):
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
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Step {st}'.format(
                           batch_time=batch_time, data_time=data_time, loss=loss_avg, st=step))

                step += 1

            write_event(log, step, loss=loss_avg.avg)
            save_checkpoint(epoch)

            valid_metrics = validation(valid_loader, model, criterion)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_metrics['score'] > best_score:
                best_score = valid_metrics['score']
                if args.best_checkpoint_metric == 'score':
                    save_best_checkpoint(valid_metrics)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if args.best_checkpoint_metric == 'valid_loss':
                    save_best_checkpoint(valid_metrics)
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
    accuracy_unmanip = np.mean(matches[unmanip_mask]) if unmanip_mask.any() else 0.
    accuracy_manip = np.mean(matches[~unmanip_mask]) if not unmanip_mask.all() else 0.
    accuracy_weighted = 0.7 * accuracy_unmanip + 0.3 * accuracy_manip
    print('Validation loss {:.4f}\t'
          'Accuracy unmanip {:.4f}\tmanip {:.4f}\tweighted: {:.4f}'.format(
        loss, accuracy_unmanip, accuracy_manip, accuracy_weighted))
    return {'valid_loss': loss, 'score': accuracy_weighted}


def predict_valid(loader: D.DataLoader, model: N.Module):
    model.eval()
    preds = []
    targets = []
    manips = []
    for batch_input, batch_targets, batch_manip in loader:
        batch_input_var = Variable(batch_input, volatile=True).cuda()
        batch_pred = model(batch_input_var)
        preds.extend(list(batch_pred.data.cpu().numpy()))
        targets.extend(list(batch_targets))
        manips.extend(list(batch_manip))

    return preds, targets, manips


def predict_test(loader: D.DataLoader, model: N.Module):
    model.eval()
    preds = []
    img_paths = []
    for batch_input, batch_img_paths in loader:
        batch_input_var = Variable(batch_input, volatile=True).cuda()
        batch_pred = model(batch_input_var)
        preds.extend(list(batch_pred.data.cpu().numpy()))
        img_paths.extend(list(batch_img_paths))

    return preds, img_paths


def save_valid_predictions(preds, targets, manips, args):
    run_dir = Path(args.run_dir)
    out_dir = Path('output') / run_dir.relative_to('.')
    if not out_dir.exists():
        os.makedirs(str(out_dir))
    save_path = out_dir / (args.mode + '.pkl')
    pickle.dump((preds, targets, manips), open(str(save_path), 'wb'))


def save_test_predictions(preds, paths, args):
    preds_cls, names = [], []
    for pred_prob, path in zip(preds, paths):
        pred_idx = np.argmax(pred_prob)
        pred_cls = dataset.IDX_TO_CLASS[pred_idx]
        name = os.path.basename(path)
        preds_cls.append(pred_cls)
        names.append(name)

    run_dir = Path(args.run_dir)
    out_dir = Path('output') / run_dir.relative_to('.')
    if not out_dir.exists():
        os.makedirs(str(out_dir))
    csv_path = out_dir / (args.mode + '.csv')
    infer_path = out_dir / (args.mode + '_detailed.pkl')
    pd.DataFrame({'fname': names, 'camera': preds_cls}, columns=['fname', 'camera']).to_csv(str(csv_path), index=False)
    pickle.dump((preds, paths), open(str(infer_path), 'wb'))


def add_arguments(parser: argparse.ArgumentParser):
    arg = parser.add_argument
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'], default='train')
    arg('-i', '--input-size', default=224, type=int, metavar='N', help='input size of the network')
    arg('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    arg('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    arg('--lr-warm', default=0.0001, type=float, metavar='LR', help='warm-up learning rate')
    arg('-r', '--run-dir', required=True, metavar='DIR', help='directory with model checkpoints, logs, etc.')
    arg('--clean', action='store_true', help='clean the output directory')
    arg('--patience', default=4, type=int, metavar='N',
        help='number of epochs without validation loss improvement to tolerate')
    arg('--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    arg('--checkpoint', choices=['best', 'last'], default='last',
        help='whether to use the best or the last model checkpoint for inference')
    arg('--best-checkpoint-metric', choices=['valid_loss', 'score'], default='score',
        help='which metric to track when saving the best model checkpoint')
    arg('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency')
    arg('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    arg('--snapshot', default='', type=str, metavar='PATH',
        help='use model snapshot to continue learning')


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    assert torch.cuda.is_available(), 'CUDA is not available'

    train_valid_transform = transforms.Compose([
        RandomRotation(),
        CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = D.ConcatDataset([
        dataset.CSVDataset(dataset.TRAINVAL_SET, transform=train_valid_transform,
                           do_manip=True, repeats=4, fix_path=utils.fix_jpg_tif),
        dataset.CSVDataset(dataset.FLICKR_TRAIN_SET, transform=train_valid_transform,
                           do_manip=True, repeats=4, fix_path=utils.fix_jpg_tif)])
    valid_dataset = dataset.CSVDataset(dataset.FLICKR_VALID_SET, transform=train_valid_transform,
                                       do_manip=True, repeats=4, fix_path=utils.fix_jpg_tif)
    test_dataset = dataset.TestDataset(transform=test_transform)

    train_loader = D.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    valid_loader = D.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = D.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = models.resnet50(num_classes=dataset.NUM_CLASSES, pretrained=True)
    #model = models.SqueezeNet(num_classes=dataset.NUM_CLASSES, pretrained=True)
    model.cuda()

    loss = N.CrossEntropyLoss()

    if 'train' == args.mode:
        if run_dir.exists() and args.clean:
            shutil.rmtree(str(run_dir))
        run_dir.mkdir(exist_ok=True)
        run_dir.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        subprocess.check_call("git diff $(find . -name '*.py') > {}".format(run_dir / 'patch'), shell=True)
        train_kwargs = {
            'args': args,
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'model': model,
            'criterion': loss,
        }

        # train(
        #     init_optimizer=lambda lr: O.SGD(model.classifier.parameters(), lr=lr, momentum=0.9),
        #     lr=args.lr_warm,
        #     n_epochs=1,
        #     **train_kwargs)
        train(
            init_optimizer=lambda lr: O.SGD([{'params': model.feature_parameters(), 'lr': lr},
                                             {'params': model.classifier_parameters(), 'lr': 1e-6}],
                                            momentum=0.9),
            lr=args.lr,
            **train_kwargs)
    elif args.mode in ['valid', 'predict_valid', 'predict_test']:
        if 'best' == args.checkpoint:
            ckpt = (run_dir / 'best-model.pt')
        else:
            ckpt = (run_dir / 'model.pt')
        state = torch.load(str(ckpt))
        model.load_state_dict(state['model'])
        print('Loaded {}'.format(ckpt))
        if 'valid' == args.mode:
            validation(tqdm.tqdm(valid_loader, desc='Validation'), model, loss)
        elif 'predict_valid' == args.mode:
            preds, targets, manips = predict_valid(valid_loader, model)
            save_valid_predictions(preds, targets, manips, args)
        elif 'predict_test' == args.mode:
            preds, paths = predict_test(test_loader, model)
            save_test_predictions(preds, paths, args)
    else:
        raise ValueError('Unknown mode {}'.format(args.mode))

if '__main__' == __name__:
    main()
