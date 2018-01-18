#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 30 Nov 2017
# Last modified : 30 Nov 2017

"""
Gaussian linear classifier on 20 newsgroup dataset
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import log_loss
from glc import GLC


def kfold_cv_dev(train_feats, train_labels, n_folds=5):
    """ Run k-fold cross validation on train set

    Args:
        train_feats (np.ndarray): training features (n_samples x dim)
        train_labels (np.ndarray): corresponding labels
        n_folds (int): number of folds (default=5)

    Returns:
        np.float64: average classification accuracy over k-folds
        np.float64: average cross-entropy loss over k-folds
    """

    skf = SKFold(n_splits=n_folds, shuffle=True, random_state=0)

    # [acc, x_entropy]
    scores = np.zeros(shape=(n_folds, 2))
    i = 0
    for trn_ixs, dev_ixs in skf.split(train_feats, train_labels):

        glc = GLC(est_prior=True)  # est class prior
        glc.train(train_feats[trn_ixs, :], train_labels[trn_ixs])

        dev_pred = glc.predict(train_feats[dev_ixs])
        dev_prob = glc.predict(train_feats[dev_ixs], return_probs=True)

        scores[i, 0] = np.mean(train_labels[dev_ixs] == dev_pred) * 100.
        scores[i, 1] = log_loss(train_labels[dev_ixs], dev_prob)
        i += 1

    return np.mean(scores[:, 0]), np.mean(scores[:, 1])


def load_data(train_npy_f):
    """ Load data """

    train_feats = np.load(train_npy_f)
    # train_labs = np.loadtxt(ARGS.train_labs_f, dtype=int)
    train_labs = np.loadtxt('20news-bydate/matlab/train.label', dtype=int)

    # test_feats = np.load(ARGS.test_npy_f)
    test_feats = np.load(ARGS.train_npy_f.replace("train", "test"))
    test_labs = np.loadtxt('20news-bydate/matlab/test.label', dtype=int)
    # test_labs = np.loadtxt(ARGS.test_labs_f, dtype=int)

    if train_feats.shape[0] != train_labs.shape[0]:
        train_feats = train_feats.T

    if test_feats.shape[0] != test_labs.shape[0]:
        test_feats = test_feats.T

    if min(train_labs) == 1:
        train_labs -= 1
    if min(test_labs) == 1:
        test_labs -= 1

    assert train_feats.shape[1] == test_feats.shape[1]

    return train_feats, train_labs, test_feats, test_labs


def main():
    """ main method """

    train_f = ARGS.train_npy_f

    max_iters = int(os.path.splitext(os.path.basename(
        ARGS.train_npy_f))[0].split("_")[-1][1:])
    print('max_iters:', max_iters)

    #  each row: dev_acc, dev_xen, test_acc, test_xen
    scores = np.zeros(shape=(max_iters, 4))
    scores[:, [1, 3]] = np.inf

    for i in range(1, max_iters+1):
        pfx = "_e" + str(max_iters)
        train_fpath = train_f.replace(pfx, "_e" + str(i))

        if not os.path.exists(train_fpath):
            continue

        train_feats, train_labs, test_feats, test_labs = load_data(train_fpath)
        scores[i-1, :2] = kfold_cv_dev(train_feats, train_labs)

        glc = GLC(est_prior=True)
        glc.train(train_feats, train_labs)
        test_pred = glc.predict(test_feats)
        test_prob = glc.predict(test_feats, return_probs=True)

        test_acc = np.mean(test_labs == test_pred) * 100.
        test_xen = log_loss(test_labs, test_prob)

        scores[i-1, 2:] = test_acc, test_xen

    dev_acc_ix = np.argmax(scores[:, 0])
    dev_xen_ix = np.argmin(scores[:, 1])
    print("                    dev_acc,      dev_xen,   test_acc,   test_xen, xtr_iter")
    print("Best dev accuracy:", scores[dev_acc_ix], dev_acc_ix+1)
    print("Best dev Xentropy:", scores[dev_xen_ix], dev_xen_ix+1)

    base = os.path.splitext(os.path.basename(ARGS.train_npy_f))[0]
    res_f = os.path.realpath(os.path.dirname(ARGS.train_npy_f) + "/../results.txt")
    header = "\ndev_acc,dev_xen,test_acc,test_xen"
    with open(res_f, 'ab') as fpw:
        np.savetxt(fpw, scores, fmt='%.4f', header=base+header)
    print("Saved to", res_f)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("train_npy_f", help="path to train feats")
    # PARSER.add_argument("train_labs_f", help="path to train labels")
    # PARSER.add_argument("test_npy_f", help="path to test feats")
    # PARSER.add_argument("test_labs_f", help="path to test labels")
    ARGS = PARSER.parse_args()
    main()
