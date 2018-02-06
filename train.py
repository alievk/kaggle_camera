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
import augmentations as aug
from augmentations import Sometimes, CenterCrop, RandomCrop, RandomRotation


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(init_optimizer, lr, n_epochs=None, lr_decay=0.2, **kwargs):
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
            metrics['valid_loss'], metrics['score']))
        shutil.copy(str(model_path), str(best_model_path))

    optimizer = init_optimizer(lr, 0)

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    log = run_dir.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    lr_changes = 0
    write_event(log, step, lr=lr)
    for epoch in range(epoch, n_epochs):
        print('Run {}\t'
              'Epoch {}/{}\t'
              'learning rate {}'.format(args.run_dir, epoch + 1, n_epochs, lr))
        model.train()
        random.seed()
        try:
            end = time.time()
            for i, (input, target, manip) in enumerate(train_loader):
                # measure data loading time
                if i > 0:  # first iteration is always slow
                    data_time.update(time.time() - end)

                input = Variable(input.float().cuda(async=True), volatile=False)
                target = Variable(target.cuda(async=True), volatile=False)
                is_manip = Variable((manip != -1).float().cuda(async=True), volatile=False)

                output = model(input, is_manip)  # type: Variable
                loss = criterion(output, target)

                optimizer.zero_grad()
                batch_size = input.size(0)
                (batch_size * loss).backward()
                optimizer.step()

                loss_avg.update(loss.data[0])
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
                if lr_changes > args.lr_max_changes:
                    print('LR changes exceeded maximum, stop.')
                    break
                print('Validation loss plateaued, decaying LR.')
                lr *= lr_decay
                lr_reset_epoch = epoch
                optimizer = init_optimizer(lr, lr_changes)
                write_event(log, step, lr=lr)
        except KeyboardInterrupt:
            print('Ctrl+C, saving snapshot')
            save_checkpoint(epoch)
            log.close()
            raise KeyboardInterrupt
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
        is_manip = Variable((manip != -1).float().cuda())

        output = model(input_var, is_manip)
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
        is_manip = Variable((batch_manip != -1).float().cuda())
        batch_pred = model(batch_input_var, is_manip)
        preds.extend(list(batch_pred.data.cpu().numpy()))
        targets.extend(list(batch_targets))
        manips.extend(list(batch_manip))

    return preds, targets, manips


def entropy(x):
    return np.sum(-x * np.log(x - x.min(axis=1, keepdims=True) + 1e-12), axis=1)


def predict_test(loader: D.DataLoader, model: N.Module, tta: bool=False):
    model.eval()
    preds = []
    img_paths = []
    for batch_input, batch_img_paths in loader:
        batch_is_manip = ['manip' in p for p in batch_img_paths]
        if tta:
            # TTA: 4 rotations
            batch_input_rot = []
            for i, b_img in enumerate(batch_input):
                for rot in [0, 1, 2, 3]:
                    batch_input_rot.append(np.rot90(b_img, rot, (1, 2)).copy())
            batch_input = torch.stack([torch.from_numpy(b) for b in batch_input_rot], 0)
            batch_is_manip = np.repeat(batch_is_manip, 4)

            batch_input_var = Variable(batch_input, volatile=True).cuda()
            batch_is_manip = Variable(torch.FloatTensor(batch_is_manip)).cuda()
            batch_pred = model(batch_input_var, batch_is_manip)

            num_aug = 4
            batch_pred = batch_pred.data.cpu().numpy()
            for i in range(batch_input.size(0) // num_aug):
                j = i * num_aug
                subbatch = batch_pred[j:j+num_aug, :]
                #pred_tta = batch_pred[j:j+num_aug, :].mean(axis=0)
                pred_tta = subbatch[np.argmin(entropy(subbatch)), :]
                preds.append(pred_tta)
        else:
            batch_input_var = Variable(batch_input, volatile=True).cuda()
            batch_is_manip = Variable(torch.FloatTensor(batch_is_manip)).cuda()
            batch_pred = model(batch_input_var, batch_is_manip)
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
    arg('-i', '--input-size', default=256, type=int, metavar='N', help='input size of the network')
    arg('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    arg('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    #arg('--lr-warm', default=0.0001, type=float, metavar='LR', help='warm-up learning rate')
    arg('--lr-max-changes', default=2, type=int, metavar='N', help='maximum number of LR changes')
    arg('-r', '--run-dir', required=True, metavar='DIR', help='directory with model checkpoints, logs, etc.')
    arg('--clean', action='store_true', help='clean the output directory')
    arg('--patience', default=4, type=int, metavar='N',
        help='number of epochs without validation loss improvement to tolerate')
    arg('--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    arg('--checkpoint', choices=['best', 'last'], default='best',
        help='whether to use the best or the last model checkpoint for inference')
    arg('--best-checkpoint-metric', choices=['valid_loss', 'score'], default='score',
        help='which metric to track when saving the best model checkpoint')
    arg('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency')
    arg('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    arg('--snapshot', default='', type=str, metavar='PATH',
        help='use model snapshot to continue learning')
    arg('--tta', action='store_true', help='do test-time augmentations')
    arg('--model', choices=['densenet121', 'densenet169', 'resnet101', 'resnet50'], default='densenet121')


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    assert torch.cuda.is_available(), 'CUDA is not available'

    train_valid_transform_1 = transforms.Compose([
        RandomCrop(args.input_size),
        RandomRotation(),  # x4
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_valid_transform_2 = transforms.Compose([
        CenterCrop(args.input_size),
        RandomRotation(),  # x4
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def init_loaders(transform):
        fix_path = utils.fix_jpg_tif
        # do_manip gives x8 samples
        train_dataset = D.ConcatDataset([
            dataset.CSVDataset(dataset.TRAINVAL_SET, args, transform=transform,
                               do_manip=True, repeats=1, fix_path=fix_path),
            dataset.CSVDataset(dataset.FLICKR_TRAIN_SET, args, transform=transform,
                               do_manip=True, repeats=1, fix_path=fix_path),
            dataset.CSVDataset(dataset.REVIEWS_SET, args, transform=transform,
                               do_manip=True, repeats=1, fix_path=fix_path)
        ])
        valid_dataset = dataset.CSVDataset(dataset.FLICKR_VALID_SET, args, transform=transform,
                                           do_manip=True, repeats=2, fix_path=fix_path)

        train_loader = D.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
        )
        valid_loader = D.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        return train_loader, valid_loader

    train_loader_1, valid_loader_1 = init_loaders(train_valid_transform_1)
    train_loader_2, valid_loader_2 = init_loaders(train_valid_transform_2)

    test_transform = transforms.Compose([
        CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = dataset.TestDataset(transform=test_transform)
    test_loader = D.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = getattr(models, args.model)(num_classes=dataset.NUM_CLASSES, pretrained=True)
    model = N.DataParallel(model).cuda()

    loss = N.CrossEntropyLoss()

    if 'train' == args.mode:
        if run_dir.exists() and args.clean:
            shutil.rmtree(str(run_dir))
        run_dir.mkdir(exist_ok=True)
        run_dir.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        subprocess.check_call("git diff $(find . -name '*.py') > {}".format(run_dir / 'patch'), shell=True)

        def init_optimizer(lr, lr_changes):
            return O.Adam(model.parameters(), lr=lr)
            # return O.SGD([{'params': model.feature_parameters(), 'lr': lr},
            #               {'params': model.classifier_parameters(), 'lr': 1e-6}], momentum=0.9)
            # if lr_changes == 0:
            #     return O.SGD([{'params': model.module.feature_parameters(), 'lr': lr},
            #                   {'params': model.module.classifier_parameters(), 'lr': 1e-6}],
            #                   momentum=0.9)
            # else:
            #     return O.SGD([{'params': model.module.feature_parameters(), 'lr': lr},
            #                   {'params': model.module.classifier_parameters(), 'lr': 1e-4}],
            #                   momentum=0.9)

        train_kwargs = {
            'args': args,
            'model': model,
            'criterion': loss,
        }

        # Train on random crops on full image
        train(
            init_optimizer=init_optimizer,
            lr=args.lr,
            train_loader=train_loader_1,
            valid_loader=valid_loader_1,
            # n_epochs=20,
            **train_kwargs)

        # Train on central crops
        # train(
        #     init_optimizer=init_optimizer,
        #     lr=args.lr,
        #     train_loader=train_loader_2,
        #     valid_loader=valid_loader_2,
        #     **train_kwargs)
    elif args.mode in ['valid', 'predict_valid', 'predict_test']:
        if 'best' == args.checkpoint:
            ckpt = (run_dir / 'best-model.pt')
        else:
            ckpt = (run_dir / 'model.pt')
        state = torch.load(str(ckpt))
        model.load_state_dict(state['model'])
        print('Loaded {}'.format(ckpt))
        if 'valid' == args.mode:
            validation(tqdm.tqdm(valid_loader_1, desc='Validation'), model, loss)
        elif 'predict_valid' == args.mode:
            preds, targets, manips = predict_valid(valid_loader_2, model)
            save_valid_predictions(preds, targets, manips, args)
        elif 'predict_test' == args.mode:
            preds, paths = predict_test(test_loader, model, args.tta)
            save_test_predictions(preds, paths, args)
    else:
        raise ValueError('Unknown mode {}'.format(args.mode))

if '__main__' == __name__:
    main()
