#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 30 Nov 2017
# Last modified : 16 Jan 2018

"""
1. Train SMM
2. Extract document i-vectors using trained model.
"""

import os
# import pwd
import sys
import argparse
import subprocess
from time import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from smm_v2 import (SMM, estimate_ubm, update_ws, update_ts,
                    update_ws_batch_wise, update_ts_batch_wise,
                    compute_loss_batch_wise)
from TwentyNewsDataset import TwentyNewsDataset
import utils


def merge_ivecs(ivecs_dir, sbase, n_batches):
    """ Merge ivec batches """

    mbase = os.path.splitext(os.path.basename(ARGS.m))[0]
    data = []
    out_f = ivecs_dir + sbase
    for i in range(ARGS.xtr+1):
        for bix in range(n_batches):
            fname = ivecs_dir + sbase + "_b" + str(bix) + "_" + mbase + "_e"
            fname += str(i+1) + ".npy"
            if os.path.exists(fname):
                data.append(np.load(fname))
                subprocess.check_call("rm -rf " + fname, shell=True)

        if len(data) > 0:
            np.save(out_f + "_" + mbase + "_e" + str(i+1) + ".npy",
                    np.concatenate(data, axis=0))
            data = []


def create_model(config, stats):
    """ Create model or load exisiting model """

    if config['latest_trn_model']:
        model = torch.load(config['latest_trn_model'])
        print("Loaded existing model:", config['latest_trn_model'])

    else:
        hyper = {'lam_w': config['lam_w'], 'reg_t': config['reg_t'],
                 'lam_t': config['lam_t'], 'iv_dim': config['iv_dim']}

        ubm = estimate_ubm(stats)
        model = SMM(stats.shape[1], ubm, hyper, config['cuda'])
        print("Model created.")

    return model


def create_sub_model(ubm, hyper, config, X):
    """ Create a sub model by considering only a subset of i-vectors """
    return SMM(X.size()[1], ubm, hyper, config['cuda'])


# @profile
def train_batch_wise(stats, config, train_loader):
    """ Train SMM batch wise (for large datasets) """

    model = create_model(config, stats)

    stime = time()

    if ARGS.trn > config['trn_done']:

        opt_w = optim.Adagrad([model.W], lr=config['eta'])
        opt_t = optim.Adagrad([model.T], lr=config['eta_t'])

        print("Training ..")
        config['trn_iters'] = ARGS.trn

        loss = compute_loss_batch_wise(model, train_loader, use='both')
        print("Init  LLH:", -loss.cpu().numpy())

        for i in range(config['trn_done'], ARGS.trn):

            loss = update_ws_batch_wise(model, opt_w, train_loader)
            loss += model.t_penalty().data.clone()
            # print("%2d W LLH: %.4f" % (i, -loss.cpu().numpy()))

            loss = compute_loss_batch_wise(model, train_loader, use='T')
            loss = update_ts_batch_wise(model, opt_t, train_loader, loss)
            loss += model.w_penalty().sum().data.clone()
            # print("%2d T LLH: %.4f" % (i, -loss.cpu().numpy()))

            if (i+1)*2 == ARGS.trn:
                print("Half-way LLH:", -loss.cpu().numpy(),
                      '%.2f' % (time() - stime))
                utils.save_model_tr(model, config, i+1)

        print("Final LLH:", -loss.cpu().numpy())
        print("Training time: %.4f sec" % (time() - stime))

        config['trn_done'] = ARGS.trn
        utils.save_model_tr(model, config, config['trn_done'])

    else:
        print("Model already exists with given number of training iterations.")
        sys.exit()

    subprocess.check_call("rm -rf " + config['tmp_dir'], shell=True)
    print("Deleted", config['tmp_dir'])


def train(stats, config, train_loader):
    """ Train SMM """

    model = create_model(config, stats)

    for data, _ in train_loader:
        if ARGS.cuda:
            data = data.cuda()
        X = Variable(data.t())

    stime = time()

    if ARGS.trn > config['trn_done']:

        opt_w = optim.Adagrad([model.W], lr=config['eta'])
        opt_t = optim.Adagrad([model.T], lr=config['eta_t'])
        print("Training ..")

        loss = model.loss(X)
        print("Init  LLH:", -loss.data.cpu().numpy())

        config['trn_iters'] = ARGS.trn

        for i in range(config['trn_done'], ARGS.trn):

            # update i-vectors W
            loss = model.loss(X, use='w')
            loss = update_ws(model, opt_w, loss, X)
            loss += model.t_penalty()
            print("W     LLH:", loss.data.cpu().numpy())

            # update bases T
            loss = model.loss(X, use='T')
            loss = update_ts(model, opt_t, loss, X)
            loss += model.w_penalty().sum()
            print("T     LLH:", loss.data.cpu().numpy())

            if (i+1)*2 == ARGS.trn:
                print("Half-way LLH:", -loss.data.cpu().numpy()[0])
                utils.save_model_tr(model, config, i+1)
        print("Final LLH:", -loss.data.cpu().numpy())
        print("Training time: %.4f sec" % (time() - stime))

        config['trn_done'] = ARGS.trn
        utils.save_model_tr(model, config, config['trn_done'])

    else:
        print("Model already exists with given number of training iterations.")
        sys.exit()

    subprocess.check_call("rm -rf " + config['tmp_dir'], shell=True)


def extract_ivectors(data, model, config, sbase):
    """ Extract i-vectors given the model and stats """

    stime = time()

    model.reset_w(data.size()[0])  # initialize i-vectors to zeros

    opt_w = optim.Adagrad([model.W], lr=config['eta'])

    mbase = os.path.splitext(os.path.basename(ARGS.m))[0]

    print("Extracting i-vectors for", data.size()[0], "docs ..")

    config['xtr_iters'] = ARGS.xtr

    if model.cuda:
        data = data.cuda()

    X = Variable(data.t())

    loss = model.loss(X)
    print("Init  LLH:", -loss.data.cpu().numpy()[0])

    for i in range(config['xtr_iters']):

        # update i-vectors W
        loss = update_ws(model, opt_w, loss, X)

        if ((i+1) % ARGS.nth == 0) or (i+1 == ARGS.xtr):
            sfx = sbase + "_" + mbase + "_e" + str(i+1) + ".npy"
            np.save(config['ivecs_dir'] + sfx, model.W.data.cpu().numpy().T)

    print("Final LLH:", -loss.data.cpu().numpy()[0])
    print("Extraction time: %.4f sec" % (time() - stime))
    print("Saved in", config['ivecs_dir'])


def main():
    """ main method """

    if ARGS.phase == 'train':
        config = utils.create_config(ARGS)
        dset = TwentyNewsDataset('train')
        train_loader = DataLoader(dset, shuffle=False, batch_size=ARGS.bs)
        stats = dset.get_data_mtx().T

        print("No. of batches:", len(train_loader))

        if len(train_loader) > 1:
            train_batch_wise(stats, config, train_loader)
        else:
            train(stats, config, train_loader)

    elif ARGS.phase == 'extract':

        if ARGS.m:
            model_f = os.path.realpath(ARGS.m)
            model, config = utils.load_model_and_config(model_f)

        else:
            print("Specify the path to trained model with option -m")
            sys.exit()

        for set_name in ['train', 'test']:
            dset = TwentyNewsDataset(set_name)
            data_loader = DataLoader(dset, shuffle=False,
                                     batch_size=ARGS.bs)
            stats = dset.get_data_mtx().T

            print("No. of batches:", len(data_loader))

            for bix, (data, _) in enumerate(data_loader):
                extract_ivectors(data, model, config,
                                 set_name + '_b' + str(bix))

            merge_ivecs(config['ivecs_dir'], set_name, len(data_loader))

    else:
        print("Invalid option. Should be train or extract.")
        sys.exit()

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    PARSER.add_argument("phase", help="train or extract")
    PARSER.add_argument("-o", default=".", help="path to output dir")
    PARSER.add_argument("-v", help="path to vocab file")
    PARSER.add_argument("-m", help="path to trained model file")
    PARSER.add_argument("-k", default=50, type=int, help="i-vector dim")
    PARSER.add_argument("-lw", default=1e-4, type=float,
                        help="reg. const. for i-vectors")
    PARSER.add_argument("-rt", default="l1", help="l1 or l2 reg. for bases T")
    PARSER.add_argument("-lt", default=1e-4, type=float,
                        help='reg. coeff. for bases T')
    PARSER.add_argument("-eta", type=float, default=0.1, help="learning rate")
    PARSER.add_argument("-eta_t", type=float, default=0.1,
                        help='learning rate for T')
    PARSER.add_argument("-bs", default=4000, type=int, help='batch size')
    PARSER.add_argument('-trn', default=50, type=int, help='training iterations')
    PARSER.add_argument('-xtr', default=20, type=int, help='extraction iterations')
    PARSER.add_argument('-mkl', default=4, type=int, help='number of MKL threads')
    PARSER.add_argument('--nth', default=5, type=int,
                        help='save every nth i-vector while extracting')
    PARSER.add_argument('--ovr', action='store_true', help='over-write the exp dir')
    PARSER.add_argument("--nocuda", action='store_true', help='Do not use GPU')

    ARGS = PARSER.parse_args()

    torch.set_num_threads(ARGS.mkl)
    torch.manual_seed(0)
    ARGS.cuda = not ARGS.nocuda and torch.cuda.is_available()
    print("CUDA:", ARGS.cuda)

    main()
