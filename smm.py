#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 21 Nov 2017
# Last modified : 21 Nov 2017

"""
Subspace Multinomial Model for learning document i-vectors
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class SMM():
    """ Subspace Multinomial Model """

    def __init__(self, stats, hyper, cuda=False):
        """ Initialize SMM

        Args:
            stats (scipy.sparse): Sparse matrix with Word-by-Doc count stats
            hyper (dict): Dictionary with hyper parameters
            cuda (boolean): Use GPU? (model is always initialized on CPU)
        """

        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        V, N = stats.shape
        self.K = hyper['iv_dim']
        self.hyper = hyper

        # universal background model
        ubm = np.log((stats.sum(axis=1) / stats.sum()).reshape(V, -1))
        self.m = Variable(torch.from_numpy(ubm).type(self.dtype),
                          requires_grad=False)

        # bases or subspace or total variability matrix
        torch.manual_seed(0)  # consistent results on CPU and GPU
        self.T = Variable(torch.randn(V, self.K).type(self.dtype),
                          requires_grad=True)

        self.W = Variable(torch.zeros(self.K, N).type(self.dtype),
                          requires_grad=True)
        # negative log-likelihood per document
        self.nllh_d = torch.zeros(N).type(self.dtype)
        # LLH over iterations
        self.llh = torch.Tensor().type(self.dtype)

    def init_w(self, N):
        """ Initialize i-vectors to zeros. """
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

    def loss(self, X):
        """ Compute loss (negative log-likelihood), given the data with
        the current model parameters.

        Args:
            X (torch.autograd.Variable): Word-by-Docs count stats

        Returns:
            loss (torch.autograd.Variable): negative LLH
        """

        mtw = (self.T @ self.W) + self.m
        log_phis = F.log_softmax(torch.t(mtw))
        w_pen = self.w_penalty()
        llh_d = torch.sum(X * torch.t(log_phis), dim=0) - w_pen
        t_pen = self.t_penalty()
        self.nllh_d = -llh_d.data.clone()
        return -(llh_d.sum() - t_pen)

    def __loss_chunk(self, X, chunk_seq):
        """ Compute loss for a chunk_seq from the data, given the
        current model parameters.

        This function is useful while backtracking for few documents.

        Args:
            X (torch.autograd.Variable): Word-by-Docs count stats.
            chunk_seq (torch.LongTensor): Seq of document indices for which
            the loss (neg. LLH) to be evaluated.

        Returns:
            loss (torch.Tensor): negative LLH

        """

        mtw = (self.T.data @ self.W.data[:, chunk_seq]) + self.m.data
        log_phis = F.log_softmax(torch.t(mtw))
        llh_chunk = torch.sum(X.data[:, chunk_seq] * torch.t(log_phis.data), dim=0)
        w_pen = self.w_penalty()

        return -(llh_chunk - w_pen.data[chunk_seq])

    def check_update_w(self, X, old_loss_d, old_w):
        """ Check if the updates have decreased the loss, else backtrack
        halving the step (max 10 steps).

        Args:
            X (torch.autograd.Variable): Variable around Word-by-Doc counts tensor
            old_loss (torch.Tensor): Tensor of dim equal to no. of docs.
            old_w (torch.Tensor): Tensor of shape (iv_dim x no. of docs.)

        Returns:
            loss (torch.autograd.Variable): negative regularized LLH tensor
        """

        loss = self.loss(X)  # get current loss (with the updated W)
        # get (doc) indices to backtrack
        bt_ixs = ((old_loss_d - self.nllh_d) < 0).nonzero().squeeze()
        bti = 0  # backtrack iters
        while bt_ixs.dim() > 0:
            self.W.data[:, bt_ixs] = (self.W.data[:, bt_ixs] +
                                      old_w[:, bt_ixs]) / 2.
            loss = self.loss(X)
            bt_ixs = ((old_loss_d - self.nllh_d) < 0).nonzero().squeeze()
            bti += 1
            if bti == 10 and bt_ixs.size()[0] > 0:
                print("BT steps > 10 for", bt_ixs.dim(), "W.")
                self.W.data[:, bt_ixs] = old_w[:, bt_ixs]  # use old_w
                loss = self.loss(X)
                break

        self.llh = torch.cat([self.llh, -loss.data])
        return loss

    def check_update_t(self, X, old_loss, old_t):
        """ Check if the updates have decreased the loss, else backtrack
        halving the step (max 10 steps).

        Args:
            X (torch.autograd.Variable): Variable around Word-by-Doc counts tensor
            old_loss (torch.Tensor): Tensor of dim equal to no. of docs.
            old_w (torch.Tensor): Tensor of shape (iv_dim x no. of docs.)

        Returns:
            loss (torch.autograd.Variable): negative regularized LLH tensor

        """

        loss = self.loss(X)
        inc = old_loss - loss.data
        bti = 0
        while (inc < 0).cpu().numpy()[0]:
            self.T.data = (self.T.data + old_t) / 2
            loss = self.loss(X)  # compute the loss again
            inc = old_loss - loss.data
            bti += 1
            if bti == 10:
                print("BT > 10 steps for T.")
                self.T.data = old_t
                loss = self.loss(X)
                break

        # self.llh = torch.cat([self.llh, -loss.data])
        return loss


def update_ws(model, opt_w, loss, X):
    """ Update i-vectors (W) """

    old_loss = model.nllh_d.clone()
    old_w = model.W.data.clone()

    opt_w.zero_grad()

    # loss = model.loss(X)
    loss.backward()

    opt_w.step()

    loss = model.check_update_w(X, old_loss, old_w)

    return loss


def update_ts(model, opt_t, loss, X):
    """ Update bases (T) """

    old_loss = loss.data.clone()
    old_t = model.T.data.clone()

    opt_t.zero_grad()

    loss.backward()

    opt_t.step()

    loss = model.check_update_t(X, old_loss, old_t)

    return loss
