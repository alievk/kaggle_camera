import argparse
from pathlib import Path
import shutil
import json

import torch
import torch.nn as N
import torch.optim as O
import torch.utils.data as D
import torchvision.models as M

import dataset
import models
import utils


def add_arguments(parser: argparse.ArgumentParser):
    arg = parser.add_argument
    arg('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    arg('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    arg('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency')
    arg('--lr', '--learning-rate', default=0.0002, type=float, metavar='LR', help='initial learning rate')
    arg('-o', '--output-dir', required=True, metavar='', help='output directory with model checkpoints, logs, etc.')
    arg('--clean', action='store_true', help='clean the output directory')
    arg('--patience', default=4, type=int, metavar='N',
        help='number of epochs without validation loss improvement to tolerate')
    arg('--epochs', default=100, type=int, metavar='N', help='number of training epochs')


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA is not available'

    train_dataset = dataset.CSVDataset(dataset.TRAIN_SET, transform=dataset.train_transform)
    valid_dataset = dataset.CSVDataset(dataset.VALID_SET, transform=dataset.valid_transform)

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

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.clean:
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(exist_ok=True)
    output_dir.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    train_kwargs = {
        'args': args,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'model': model,
        'criterion': loss,
    }

    utils.train(
        init_optimizer=lambda lr: O.SGD(model.fresh_parameters(), lr=lr, momentum=0.9),
        n_epochs=1,
        **train_kwargs)

    utils.train(
        init_optimizer=lambda lr: O.SGD(model.parameters(), lr=lr, momentum=0.9),
        **train_kwargs)

if '__main__' == __name__:
    main()
