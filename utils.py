#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 10 Dec 2017
# Last modified : 10 Dec 2017

"""
Common util functions
"""

import os
import sys
import json
import codecs
import subprocess
import tempfile
from collections import OrderedDict
import torch
import scipy.io as sio


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
        config['tmp_dir'] = tempfile.mkdtemp() + "/"
        print("Tmp dir:", config['tmp_dir'])

        try:
            config['stats_file'] = os.path.realpath(args.stats_file)
        except AttributeError:
            pass

        if args.v:
            config['vocab_file'] = os.path.realpath(args.v)

        config['iv_dim'] = args.k
        config['lam_w'] = args.lw  # prior (precision) for ivecs
        config['reg_t'] = args.rt.lower()
        config['lam_t'] = args.lt
        config['b_size'] = args.bs
        config['eta'] = args.eta
        config['eta_t'] = args.eta_t
        config['cuda'] = args.cuda

        config['trn_iters'] = args.trn
        config['xtr_iters'] = args.xtr

        # useful to continue training or extracting from the latest model
        config['latest_trn_model'] = None
        config['latest_xtr_model'] = None

        config['trn_done'] = 0
        config['xtr_done'] = 0

        print("Config file created.")
        json.dump(config, open(cfg_file, "w"), indent=2, sort_keys=True)

    return config


def load_sub_model(config, sfx):
    """ Load sub model """
    model_f = config['tmp_dir'] + "sub_model_T" + sfx + ".pt"
    return torch.load(model_f)


def save_sub_model(config, model, sfx):
    """ Save sub model to temp dir """
    model_f = config['tmp_dir'] + "sub_model_T" + sfx + ".pt"
    torch.save(model, model_f)


def save_model_tr(model, config, itr):
    """ Save model and config """

    sfx = str(itr) + '.pt'
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
        print("Loaded:", model_f)
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
