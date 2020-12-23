import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self,
                 q_encoder,
                 h_dim=128,
                 z_dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07,
                 distributed=False):
        """
        h_dim: latent space dimension (default: 128)
        z_dim: projection head dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.distributed = distributed

        # create the encoders for MoCo and surpervised model
        self.encoder_q = q_encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # create the projection mlp layers for MoCo
        self.mlp_q = nn.Sequential(nn.Linear(1024, h_dim), nn.ReLU(),
                                   nn.Linear(h_dim, h_dim), nn.ReLU(),
                                   nn.Linear(h_dim, z_dim))
        self.mlp_k = nn.Sequential(nn.Linear(1024, h_dim), nn.ReLU(),
                                   nn.Linear(h_dim, h_dim), nn.ReLU(),
                                   nn.Linear(h_dim, z_dim))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # initialize encoder_k and set to not update by gradient
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # initialize mlp_k and set to not update by gradient
        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
 
        # create the queue
        self.register_buffer("queue", torch.randn(z_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder and the key mlp
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if self.distributed:
            # gather keys before updating queue
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # encode query features
        h_q = self.encoder_q(im_q)['res4']
        h_q = self.avgpool(h_q)
        h_q = torch.flatten(h_q, 1)
        z_q = self.mlp_q(h_q)  # NxC
        z_q = nn.functional.normalize(z_q, dim=1)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            if self.distributed:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            h_k = self.encoder_k(im_k)['res4']
            h_k = self.avgpool(h_k)
            h_k = torch.flatten(h_k, 1)
            z_k = self.mlp_k(h_k)  # NxC
            z_k = nn.functional.normalize(z_k, dim=1)

            if self.distributed:
                # undo shuffle
                z_k = self._batch_unshuffle_ddp(z_k, idx_unshuffle)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)  # Nx1
        l_neg = torch.einsum('nc,ck->nk',
                             [z_q, self.queue.clone().detach()])  # NxK
        logits = torch.cat([l_pos, l_neg], dim=1)  # Nx(1+K)
        logits /= self.T  # apply temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(z_k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class base_encoder(nn.Module):
    """
    Base encoder for the MoCo and supervised learning
    Architecture from DexNet 2.0
    Input size 32*32*1 (grey depth image)
    """
    def __init__(self, num_classes=128):
        super(base_encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 7)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        # self.fc = nn.Linear(64 * 7 * 7, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.lrn = nn.LocalResponseNorm(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(self.pool(x)))
        x = self.lrn(F.relu(self.conv4(x)))
        # x = x.view(-1, 64 * 7 * 7)
        # x = self.fc(x)
        return x


class CG(nn.Module):
    """
    Build supervised learning model head using encoder from moco
    """
    def __init__(self, moco, h_dim=128):
        super(CG, self).__init__()
        self.encoder = moco.encoder_q
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
