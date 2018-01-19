#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju, Lukas Burget, Mehdi Soufifar
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 21 Nov 2017
# Last modified : 03 Dec 2017

"""
Subspace Multinomial Model for learning document representations (i-vectors)
Paper:
http://www.fit.vutbr.cz/research/groups/speech/publi/2016/kesiraju_interspeech2016_IS161634.pdf
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def estimate_ubm(stats):
    """ Given the stats (scipy.sparse), estimate UBM (ML) """
    # universal background model or log-average dist. over vocabulary
    return torch.from_numpy(np.log((stats.sum(axis=1) /
                                    stats.sum()).reshape(-1, 1))).float()


class SMM():
    """ Subspace Multinomial Model """

    def __init__(self, N, ubm, hyper, cuda=False):
        """ Initialize SMM

        Args:
            N (int): number of docs
            ubm (torch.Tensor): Universal backgroud model
            hyper (dict): Dictionary with hyper parameters
            cuda (boolean): Use GPU? (model is always initialized on CPU)
        """

        # import pdb
        # pdb.set_trace()

        self.cuda = cuda
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        V = ubm.size()[0]
        self.K = hyper['iv_dim']
        self.hyper = hyper

        self.m = Variable(ubm.type(self.dtype), requires_grad=False)

        torch.manual_seed(0)  # for consistent results on CPU and GPU
        # bases or subspace or total variability matrix
        self.T = Variable(torch.randn(V, self.K).type(self.dtype),
                          requires_grad=True)

        # i-vectors
        self.W = Variable(torch.zeros(self.K, N).type(self.dtype),
                          requires_grad=True)

        # negative log-likelihood per document
        self.nllh_d = torch.zeros(N).type(self.dtype)
        # LLH over iterations
        self.llh = torch.Tensor().type(self.dtype)

    def reset_w(self, N):
        """ Initialize (reset) i-vectors to zeros. """
        # i-vectors
        self.W = Variable(torch.zeros(self.K, N).type(self.dtype),
                          requires_grad=True)
        # negative log-likelihood per document
        self.nllh_d = torch.zeros(N).type(self.dtype)
        # LLH over iterations
        self.llh = torch.Tensor().type(self.dtype)

    def t_penalty(self):
        """ Compute penalty term (regularization) for the bases """

        lam_t = Variable(torch.Tensor([self.hyper['lam_t']]).type(self.dtype))
        if self.hyper['reg_t'] == 'l2':
            t_pen = lam_t * torch.sum(torch.pow(self.T, 2))
        else:
            t_pen = lam_t * torch.sum(torch.abs(self.T.data))

        return t_pen

    def w_penalty(self):
        """ Compute penalty term (regularization) for the i-vectors """

        lam_w = Variable(torch.Tensor([self.hyper['lam_w']]).type(self.dtype))
        w_pen = lam_w * torch.sum(torch.pow(self.W, 2), dim=0)
        return w_pen

    def loss(self, X, use='both'):
        """ Compute loss (negative log-likelihood), given the data with
        the current model parameters.

        Args:
            X (torch.autograd.Variable): Word-by-Docs count stats
            use (str): Params to use to compute the loss, W or T or both

        Returns:
            loss (torch.autograd.Variable): negative LLH
        """

        mtw = (self.T @ self.W) + self.m
        log_phis = F.log_softmax(mtw, dim=0)
        llh_d = torch.sum((X * log_phis), dim=0)

        w_pen = self.w_penalty()
        t_pen = self.t_penalty()

        if use == 'w':
            llh_d -= w_pen
            self.nllh_d = -llh_d.data.clone()
            loss = -llh_d.sum()

        elif use == 'T':
            loss = -(llh_d.sum() - t_pen)

        elif use == 'both':
            llh_d -= w_pen
            self.nllh_d = -llh_d.data.clone()
            loss = -(llh_d.sum() - t_pen)

        return loss

    def loss_batch(self, X, rng, use='both'):
        """ Compute loss for a batch of data, using the corresponding batch
        of current model parameters.

        Args:
            X (torch.autograd.Variable): Batch of Word-by-Docs count stats.
            rng (tuple): start and end of batch
            use (str): w or None

        Returns:
            loss_batch (torch.autograd.Variable): negative LLH for a batch

        """

        mtw = (self.T @ self.W[:, rng[0]:rng[1]]) + self.m
        log_phis = F.log_softmax(mtw, dim=0)
        llh_batch = torch.sum(X * log_phis, dim=0)

        if use.lower() == 'w' or use.lower() == 'both':
            w_pen = self.w_penalty()
            self.nllh_d[rng[0]:rng[1]] = -(llh_batch -
                                           w_pen[rng[0]:rng[1]]).data.clone()
            loss_batch = -torch.sum(llh_batch - w_pen[rng[0]:rng[1]])

        else:
            loss_batch = -torch.sum(llh_batch)

        return loss_batch


def update_ws(model, opt_w, loss, X, rng=None):
    """ Update i-vectors (W) """

    old_loss_d = model.nllh_d.clone()
    old_w = model.W.data.clone()

    opt_w.zero_grad()
    loss.backward()
    opt_w.step()

    # Check if the updates have decreased the loss, else backtrack
    # halving the step (max 10 steps).

    if rng:
        loss = model.loss_batch(X, rng, use='w')
    else:
        loss = model.loss(X, use='w')  # get current loss (with the updated W)

    # get (doc) indices to backtrack
    bt_ixs = ((old_loss_d - model.nllh_d) < 0).nonzero().squeeze()
    bti = 0  # backtrack iters
    while bt_ixs.dim() > 0:
        model.W.data[:, bt_ixs] = (model.W.data[:, bt_ixs] + old_w[:, bt_ixs]) / 2.
        if rng:
            loss = model.loss_batch(X, rng, use='w')
        else:
            loss = model.loss(X, use='w')  # get current loss (with the updated W)

        bt_ixs = ((old_loss_d - model.nllh_d) < 0).nonzero().squeeze()
        bti += 1
        if bti == 10 and bt_ixs.dim() > 0:
            print("BT steps > 10 for", bt_ixs.dim(), "W.")
            model.W.data[:, bt_ixs] = old_w[:, bt_ixs]  # use old_w

            if rng:
                loss = model.loss_batch(X, rng, use='w')
            else:
                loss = model.loss(X, use='w')

            break

    return loss


def sign_projection(model, config):
    """ Sign projection in case of L1 regularization """

    T = model.T.data.cpu().clone().numpy()
    grad = model.T.grad.data.cpu().clone().numpy()

    diff_pts = T.nonzero()
    grad[diff_pts] += (config['lam_t'] * np.sign(T[diff_pts]))

    # sub-gradients
    non_diff_pts = np.where(T == 0)
    if len(non_diff_pts[0]) > 0:
        for row, col in zip(non_diff_pts[0], non_diff_pts[1]):
            if grad[row, col] < -config['lam_t']:
                grad[row, col] += config['lam_t']
            elif grad[row, col] > config['lam_t']:
                grad[row, col] -= config['lam_t']
            elif abs(grad[row, col]) <= config['lam_t']:
                grad[row, col] = 0.
            else:
                continue

    if model.cuda:
        model.T.grad.data = torch.from_numpy(grad).float().cuda()
    else:
        model.T.grad.data = torch.from_numpy(grad).float()


def update_ts(model, opt_t, loss, X, config):
    """ Update bases (T) """

    old_loss = loss.data.clone()
    old_t = model.T.data.clone()

    opt_t.zero_grad()
    loss.backward()  # get the gradients

    if config['reg_t'] == 'l1':
        sign_projection(model, config)

    opt_t.step()

    # Check if the updates have decreased the loss, else backtrack
    # halving the step (max 10 steps).

    loss = model.loss(X, use='T')
    inc = old_loss - loss.data
    bti = 0
    while (inc < 0).cpu().numpy()[0]:
        model.T.data = (model.T.data + old_t) / 2
        loss = model.loss(X, use='T')  # compute the loss again
        inc = old_loss - loss.data
        bti += 1
        if bti == 10:
            print("BT > 10 steps for T.")
            model.T.data = old_t
            loss = model.loss(X, use='T')
            break

    # model.llh = torch.cat([model.llh, -loss.data])
    return loss


def compute_loss_batch_wise(model, data_loader, use='both'):
    """ Compute total loss batch-wise.

    Args:
        model (SMM object):
        data_loader (torch.utils.data DataLoader object):
        use (str): w or T or both

    Returns:
        loss (torch.Tensor): negative LLH
    """

    rng = [0, 0]
    loss = torch.Tensor([0]).type(model.dtype)

    for data, _ in data_loader:

        if model.cuda:
            data = data.cuda()

        X = Variable(data.t())
        rng = [rng[1], rng[1] + X.size()[1]]

        loss_batch = model.loss_batch(X, rng, use=use)
        loss += loss_batch.data

    if use == 'T' or use == 'both':
        t_pen = model.t_penalty()
        loss += t_pen.data.clone()

    return loss


def update_ws_batch_wise(model, opt_w, data_loader):
    """ Update i-vectors (w) batch wise

    Args:
        model (SMM object):
        opt_w (torch.optim.Adagrad object):
        data_loader (torch.utils.data DataLoader object):

    Returns:
        loss (torch.Tensor): negative LLH

    """

    rng = [0, 0]
    loss = torch.Tensor([0]).type(model.dtype)
    for data, _ in data_loader:

        if model.cuda:
            data = data.cuda()
        X = Variable(data.t())

        rng = [rng[1], rng[1] + X.size()[1]]

        loss_batch = model.loss_batch(X, rng, 'w')
        loss_batch = update_ws(model, opt_w, loss_batch, X, rng)
        loss += loss_batch.data

    return loss


def update_ts_batch_wise(model, opt_t, data_loader, old_loss, config):
    """ Update bases batch wise

    Args:
        model (SMM object):
        opt_t (torch.optim.Adagrad object):
        data_loader (torch.utils.data DataLoader object):
        old_loss (torch.Tensor): Old loss w.r.t. `T'
        config (dict): Configuration dict

    Returns:
        loss (torch.Tensor): negative LLH

    """

    old_t = model.T.data.clone()
    opt_t.zero_grad()

    rng = [0, 0]  # range = docs. start and end indices
    loss = torch.Tensor([0]).type(model.dtype)

    for bix, (data, _) in enumerate(data_loader):

        if model.cuda:
            data = data.cuda()

        X = Variable(data.t())
        rng = [rng[1], rng[1] + X.size()[1]]

        loss_batch = model.loss_batch(X, rng, 'T')
        if bix < len(data_loader) - 1:
            loss_batch.backward()

        else:
            # for the last batch, add T penalty
            t_pen = model.t_penalty()
            loss_batch += t_pen
            loss_batch.backward()

            if config['reg_t'] == 'l1':
                sign_projection(model, config)

            opt_t.step()

            loss = compute_loss_batch_wise(model, data_loader, use='T')

            # check if the updates decreased the loss
            inc = old_loss - loss
            bti = 0
            while (inc < 0).cpu().numpy()[0]:
                model.T.data = (model.T.data + old_t) / 2.
                # compute the loss after halving the step
                loss = compute_loss_batch_wise(model, data_loader, use='T')
                inc = old_loss - loss
                bti += 1
                if bti == 10:
                    print("BT > 10 steps for T.")
                    model.T.data = old_t
                    loss = compute_loss_batch_wise(model, data_loader, use='T')
                    break

    model.llh = torch.cat([model.llh, -loss])
    return loss
