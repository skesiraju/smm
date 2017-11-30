#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 30 Nov 2017
# Last modified : 30 Nov 2017

"""
Train SMM or extract document i-vectors using existing trained model.
"""

import os
# import pwd
import sys
import json
import codecs
import argparse
import subprocess
from time import time
from collections import OrderedDict
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from torch.autograd import Variable
from smm import SMM


def read_simple_flist(fname):
    """ Load a file into list. Should be called from smaller files only. """

    lst = []
    with codecs.open(fname, 'r') as fpr:
        lst = fpr.readlines()
    if lst[-1].strip() == '':
        lst = lst[:-1]
    return lst


def create_config(args):
    """ Create configuration """

    exp_dir = os.path.realpath(args.o) + "/"
    exp_dir += "lw_{:.0e}_{:s}_{:.0e}".format(args.lw, args.rt, args.lt)
    exp_dir += "_{:d}".format(args.k) + "/"

    if args.ovr:
        if os.path.exists(exp_dir):
            print('Overwriting existing output dir:', exp_dir)
            subprocess.check_call("rm -rf " + exp_dir, shell=True)
    os.makedirs(exp_dir, exist_ok=True)

    cfg_file = exp_dir + "config.json"
    config = OrderedDict()

    try:
        config = json.load(open(cfg_file, 'r'))
        print('Config:', cfg_file, 'loaded.')

    except IOError:

        ivecs_d = exp_dir + "ivecs/"
        os.makedirs(ivecs_d, exist_ok=True)

        config['cfg_file'] = cfg_file  # this file
        config['exp_dir'] = exp_dir
        config['ivecs_dir'] = ivecs_d

        config['stats_file'] = os.path.realpath(args.stats_file)
        if args.v:
            config['vocab_file'] = os.path.realpath(args.vocab_file)

        config['iv_dim'] = args.k
        config['lam_w'] = args.lw  # prior (precision) for ivecs
        config['reg_t'] = args.rt
        config['lam_t'] = args.lt
        config['b_size'] = args.bs
        config['eta'] = args.eta

        config['trn_iters'] = args.trn
        config['xtr_iters'] = args.xtr

        # useful to continue training or extracting from the latest model
        config['latest_trn_model'] = None
        config['latest_xtr_model'] = None

        config['trn_done'] = 0
        config['xtr_done'] = 0
        config['n_chunks'] = args.nc

        print("Config file created.")
        json.dump(config, open(cfg_file, "w"), indent=2, sort_keys=True)

    return config

def create_model(config, stats):
    """ Create model or load exisiting model """

    if config['latest_trn_model']:
        model = load_model_and_config(config['latest_trn_model'])
        print("Loaded existing model:", config['latest_trn_model'])

    else:
        hyper = {'lam_w': config['lam_w'], 'reg_t': config['reg_t'],
                 'lam_t': config['lam_t'], 'iv_dim': config['iv_dim']}
        model = SMM(stats, hyper)
        print("Model created.")

    return model


def save_model_tr(model, config, sfx):
    """ Save model and config """

    config['latest_trn_model'] = config['exp_dir'] + 'model_T' + sfx
    torch.save(model, config['latest_trn_model'])
    json.dump(config, open(config['cfg_file'], 'w'), indent=2)
    print("Model saved:", config['latest_trn_model'])


def load_model_and_config(model_f):
    """ Load existing model and config """

    try:
        model = torch.load(model_f)
        cfg_file = os.path.dirname(os.path.realpath(model_f)) + "/config.json"
        config = json.load(open(cfg_file, 'r'))
    except IOError as err:
        print("Cannot load model:", model_f)
        print(err)
        sys.exit()

    return model, config


def load_stats(stats_f, vocab_f):
    """ Validate and load the input stats """

    stats = sio.mmread(stats_f)

    if vocab_f:
        vocab = read_simple_flist(vocab_f)

        # Check the compatibility of stats
        if stats.shape[1] == len(vocab):
            stats = stats.T
            print("Transposed the stats to make them word-by-doc.")
            sio.mmwrite(os.path.realpath(stats_f), stats)

        if stats.shape[0] != len(vocab):
            print("Number of rows in stats should match with length of vocabulary.")
            print("Given stats:", stats.shape[0], "vocab. length:", len(vocab))
            sys.exit()

    return stats.tocsc()

def train(stats, config):
    """ Train SMM """

    model = create_model(config, stats)

    stime = time()

    if ARGS.trn > config['trn_done']:

        opt_w = optim.Adagrad([model.W], lr=config['eta'])
        opt_t = optim.Adagrad([model.T], lr=config['eta'])

        print("Training ..")

        config['trn_iters'] = ARGS.trn

        X = Variable(torch.from_numpy(stats.A.astype(np.float32)).float())
        loss = model.loss(X)
        print("Init  LLH:", -loss.data.numpy()[0])

        for i in range(config['trn_done'], ARGS.trn):

            # update i-vectors W
            old_loss = model.nllh_d.data.clone()
            old_w = model.W.data.clone()
            opt_w.zero_grad()

            loss = model.loss(X)
            loss.backward()
            opt_w.step()
            loss = model.check_update_w(X, old_loss, old_w)

            # update bases T
            old_loss = loss.data.clone()
            old_t = model.T.data.clone()
            opt_t.zero_grad()

            loss.backward()
            opt_t.step()
            loss = model.check_update_t(X, old_loss, old_t)

        print("Final LLH:", -loss.data.numpy()[0])
        print("Training time: %.4f sec" % (time() - stime))

        config['trn_done'] = ARGS.trn
        sfx = str(config['trn_done']) + '.pt'
        save_model_tr(model, config, sfx)

    else:
        print("Model already exists with given number of training iterations.")
        sys.exit()


def extract_ivectors(stats, model, config):
    """ Extract i-vectors given the model and stats """

    stime = time()

    model.init_w(stats.shape[1])  # initialize i-vectors to zeros
    opt_w = optim.Adagrad([model.W], lr=config['eta'])

    sbase = os.path.splitext(os.path.basename(ARGS.stats_file))[0]
    mbase = os.path.splitext(os.path.basename(ARGS.m))[0]

    print("Extracting i-vectors for", stats.shape[1], "docs ..")

    config['xtr_iters'] = ARGS.xtr

    X = Variable(torch.from_numpy(stats.A.astype(np.float32)).float())
    loss = model.loss(X)
    print("Init  LLH:", -loss.data.numpy()[0])

    for i in range(config['xtr_iters']):

        # update i-vectors W
        old_loss = model.nllh_d.data.clone()
        old_w = model.W.data.clone()
        opt_w.zero_grad()

        loss = model.loss(X)
        loss.backward()
        opt_w.step()
        loss = model.check_update_w(X, old_loss, old_w)

        if ((i+1) % ARGS.nth == 0) or (i+1 == ARGS.xtr):
            sfx = sbase + "_" + mbase + "_e" + str(i+1) + ".npy"
            np.save(config['ivecs_dir'] + sfx, model.W.data.numpy().T)

    print("Final LLH:", -loss.data.numpy()[0])
    print("Extraction time: %.4f sec" % (time() - stime))
    print("Saved in", config['ivecs_dir'])


def main():
    """ main method """

    if ARGS.phase == 'train':
        config = create_config(ARGS)
        stats = load_stats(ARGS.stats_file, ARGS.v)
        train(stats, config)

    elif ARGS.phase == 'extract':
        stats = load_stats(ARGS.stats_file, ARGS.v)

        if ARGS.m:
            model_f = os.path.realpath(ARGS.m)
            model, config = load_model_and_config(model_f)
            extract_ivectors(stats, model, config)

        else:
            print("Specify the path to trained model with option -m")
            sys.exit()

    else:
        print("Invalid option. Should be train or extract.")
        sys.exit()

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    PARSER.add_argument("phase", help="train or extract")
    PARSER.add_argument("stats_file",
                        help="scipy.sparse stats_file.mtx in Word x Docs shape")
    PARSER.add_argument("-o", default=".", help="path to output dir")
    PARSER.add_argument("-v", help="path to vocab file")
    PARSER.add_argument("-m", help="path to trained model file")
    PARSER.add_argument("-k", default=50, type=int, help="ivector dim")
    PARSER.add_argument("-lw", default=1e-4, type=float,
                        help="reg. const. for i-vecs")
    PARSER.add_argument("-rt", default="l1", help="l1 or l2 reg. for bases T")
    PARSER.add_argument("-lt", default=1e-4, type=float,
                        help='reg. coeff. for bases T')
    PARSER.add_argument("-eta", type=float, default=0.1, help="learning rate")
    PARSER.add_argument("-bs", default=5000, type=int, help='batch size')
    PARSER.add_argument("-nc", default=5, type=int, help="number of chunks")
    PARSER.add_argument('-trn', default=50, type=int, help='training iterations')
    PARSER.add_argument('-xtr', default=20, type=int, help='extraction iterations')
    PARSER.add_argument('-mkl', default=4, type=int, help='number of MKL threads')
    PARSER.add_argument('--nth', default=5, type=int,
                        help='save every nth i-vector while extracting')
    PARSER.add_argument('--ovr', action='store_true', help='over-write the exp dir')

    ARGS = PARSER.parse_args()

    torch.set_num_threads(ARGS.mkl)
    torch.manual_seed(0)

    main()
