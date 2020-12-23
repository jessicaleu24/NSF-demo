import argparse
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_1D import resnet18, resnet34, resnet50  # modified for 1 channel depth images (conv1)
from model.cg_model import base_encoder, MoCo, CG

parser = argparse.ArgumentParser(
    description='PyTorch Contrastive Grasping Training')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--mlr',
                    '--moco-learning-rate',
                    default=0.03,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='mlr')
parser.add_argument('--clr',
                    '--cg-learning-rate',
                    default=0.03,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='clr')
parser.add_argument('--moco_schedule',
                    default=[120, 160],
                    nargs='*',
                    type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cg_schedule',
                    default=[120, 160],
                    nargs='*',
                    type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--moco_momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--cg_momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--mwd',
                    '--moco_weight_decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='moco_weight_decay')
parser.add_argument('--cwd',
                    '--cg_weight_decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='cg_weight_decay')

parser.add_argument('--enocoder_dim',
                    default=128,
                    type=int,
                    help='encoder dimension (default: 128)')
parser.add_argument('--moco_dim',
                    default=128,
                    type=int,
                    help='moco dimension (default: 128)')
parser.add_argument(
    '--moco_k',
    default=65536,
    type=int,
    help='queue size; number of negative keys (default: 65536)')
parser.add_argument(
    '--moco_m',
    default=0.999,
    type=float,
    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco_t',
                    default=0.07,
                    type=float,
                    help='softmax temperature (default: 0.07)')

if __name__ == '__main__':
    args = parser.parse_args()

    moco = MoCo(resnet18,
                h_dim=args.enocoder_dim,
                z_dim=args.moco_dim,
                K=args.moco_k,
                m=args.moco_m,
                T=args.moco_t,
                distributed=False)
    cg = CG(moco, h_dim=args.enocoder_dim)

    moco.cuda()
    cg.cuda()
    # moco = torch.nn.parallel.DistributedDataParallel(moco)

    moco_optimizer = torch.optim.SGD(moco.parameters(),
                                     args.mlr,
                                     momentum=args.moco_momentum,
                                     weight_decay=args.moco_weight_decay)
    cg_optimizer = torch.optim.SGD(cg.parameters(),
                                   args.clr,
                                   momentum=args.cg_momentum,
                                   weight_decay=args.cg_weight_decay)
    moco_criterion = nn.CrossEntropyLoss().cuda()
    cg_criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(args.epochs):
        moco.train()
        cg.train()

        im_q = torch.randn(2, 1, 512, 512).cuda(non_blocking=True)
        im_k = torch.randn(2, 1, 512, 512).cuda(non_blocking=True)
        # may need to normalize the input image
        # im_q = nn.functional.normalize(im_q, dim=1)
        # im_k = nn.functional.normalize(im_k, dim=1)

        logits, labels = moco(im_q, im_k)

        moco_loss = moco_criterion(logits, labels)
        moco_optimizer.zero_grad()
        moco_loss.backward()
        moco_optimizer.step()

        pred = cg(im_q)
        labels = torch.randint(0, 2, (2, )).cuda(non_blocking=True)
        cg_loss = cg_criterion(pred, labels)
        cg_optimizer.zero_grad()
        cg_loss.backward()
        cg_optimizer.step()

        print("Done")