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
import argparse
import h5py
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

        scores[i, :2] = run_glc(train_feats[trn_ixs], train_labels[trn_ixs],
                                train_feats[dev_ixs], train_labels[dev_ixs])
        i += 1

    return np.mean(scores[:, 0]), np.mean(scores[:, 1])


def run_glc(train_feats, train_labels, test_feats, test_labels):
    """ Train and classify using Gaussian linear classifier """

    glc = GLC(est_prior=True)
    glc.train(train_feats, train_labels)
    test_pred = glc.predict(test_feats)
    test_prob = glc.predict(test_feats, return_probs=True)

    test_acc = np.mean(test_labels == test_pred) * 100.
    test_xen = log_loss(test_labels, test_prob)

    return test_acc, test_xen


def run(train_h5, test_h5, max_iters, mbase):
    """ Train and classify for every iteration of extracted i-vector """

    #  each row: dev_acc, dev_xen, test_acc, test_xen
    scores = np.zeros(shape=(max_iters, 4))
    scores[:, [1, 3]] = np.inf

    train_labels = np.loadtxt('20news-bydate/matlab/train.label', dtype=int)
    test_labels = np.loadtxt('20news-bydate/matlab/test.label', dtype=int)

    if min(train_labels) == 1:
        train_labels -= 1
    if min(test_labels) == 1:
        test_labels -= 1

    for i in range(1, max_iters+1):

        train_f = TRN_BASE + "_model_" + mbase + "_e" + str(i)
        test_f = train_f.replace(TRN_BASE, TEST_BASE)

        try:

            train_feats = train_h5.get(train_f).value
            test_feats = test_h5.get(test_f).value

            if train_feats.shape[0] != train_labels.shape[0]:
                train_feats = train_feats.T

            if test_feats.shape[0] != test_labels.shape[0]:
                test_feats = test_feats.T

            scores[i-1, :2] = kfold_cv_dev(train_feats, train_labels)

            scores[i-1, 2:] = run_glc(train_feats, train_labels, test_feats, test_labels)

        except AttributeError:
            pass

    return scores


def main():
    """ main method """

    ivecs_h5f = os.path.realpath(ARGS.ivecs_h5)
    h5f = h5py.File(ivecs_h5f, 'r')
    train_h5 = h5f.get('train')
    test_h5 = h5f.get('test')

    max_iters = int(os.path.splitext(os.path.basename(
        ARGS.ivecs_h5))[0].split("_")[-1][1:])
    print('max_iters:', max_iters)
    mbase = os.path.splitext(os.path.basename(ivecs_h5f))[0].split("_")[-2]
    print('mbase:', mbase)

    scores = run(train_h5, test_h5, max_iters, mbase)

    dev_acc_ix = np.argmax(scores[:, 0])
    dev_xen_ix = np.argmin(scores[:, 1])
    print("                    dev_acc,      dev_xen,   test_acc,   test_xen, xtr_iter")
    print("Best dev accuracy:", scores[dev_acc_ix], dev_acc_ix+1)
    print("Best dev Xentropy:", scores[dev_xen_ix], dev_xen_ix+1)

    base = os.path.splitext(os.path.basename(ARGS.ivecs_h5))[0]
    res_f = os.path.realpath(os.path.dirname(ARGS.ivecs_h5) + "/../results.txt")
    header = "\ndev_acc,dev_xen,test_acc,test_xen"
    if ARGS.ovr:
        mode = 'wb'
    else:
        mode = 'ab'
    with open(res_f, mode) as fpw:
        np.savetxt(fpw, scores, fmt='%.4f', header=base+header)
    print("Saved to", res_f)

    with open(res_f.replace('results.txt', 'best_score.txt'), 'w') as fpw:
        np.savetxt(fpw, scores[dev_acc_ix], fmt='%.4f', header=header[1:])


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("ivecs_h5", help="path to ivecs.h5 file")
    PARSER.add_argument("--ovr", action="store_true",
                        help="over-write results file")
    ARGS = PARSER.parse_args()

    TRN_BASE = 'train'
    TEST_BASE = 'test'

    main()
