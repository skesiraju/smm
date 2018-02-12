#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 03 Dec 2017
# Last modified : 03 Dec 2017

"""
20 Newsgroup dataset
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import scipy.sparse


class TwentyNewsDataset(Dataset):
    """ 20 Newsgroup dataset """

    def __init__(self, set_name):
        """ init """

        data_dir = '20news-bydate/'        
        self.data_mtx = sio.mmread(data_dir + set_name + '.mtx').tocsc()
        labs = np.loadtxt(data_dir + 'matlab/' + set_name + '.label', dtype=int)

        if labs.shape[0] == self.data_mtx.shape[1]:
            self.data_mtx = self.data_mtx.T

        self.data = torch.from_numpy(self.data_mtx.A.astype(np.float32))
        self.labs = torch.from_numpy(labs)

    def __getitem__(self, idx):
        return self.data[idx, :], self.labs[idx]

    def __len__(self):
        return self.data.shape[0]

    def get_data_mtx(self):
        """ Return stats in scipy.sparse.csc format """
        return self.data_mtx.tocsc()

    def get_labels(self):
        """ Return labels in numpy ndarray format """
        return self.labs.numpy()


def download_data():
    """ Download the pre-processed 20Newsgroup data from online """

    data_url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz'
    vocab_url = 'http://qwone.com/~jason/20Newsgroups/vocabulary.txt'

    data_f = data_url.split("/")[-1]
    if not os.path.exists(data_f):
        os.system("wget " + data_url)

    vocab_f = vocab_url.split("/")[-1]
    if not os.path.exists(vocab_f):
        os.system("wget " + vocab_url)

    os.system("tar -xvf " + data_f)
    os.system("mv " + vocab_f + " 20news-bydate/")


def get_sparse_matrix(fname):
    """ Read the file in matlab format and return scipy.sparse.coo matrix.

    http://qwone.com/~jason/20Newsgroups/
    The .data files are formatted "docIdx wordIdx count".
    The .label files are simply a list of label id's
    """

    print("Converting", fname, "..")
    # format: docID wordID cnt
    data = np.loadtxt(fname, dtype=int)
    rows = data[:, 0] - 1  # needed because we want the indices to start from 0
    cols = data[:, 1] - 1  # needed because we want the indices to start from 0
    vals = data[:, 2]

    print('max doc ix:', max(rows), 'max word ix:', max(cols))

    coo = scipy.sparse.coo_matrix((vals, (rows, cols)))

    return coo

def main():
    """ main method """

    data_d = '20news-bydate'
    if not os.path.isdir(data_d):
        download_data()

    trn_f = "20news-bydate/matlab/train.data"
    test_f = "20news-bydate/matlab/test.data"

    if ((not os.path.exists(data_d + '/train.mtx')) or
            (not os.path.exists(data_d + '/test.mtx'))):

        trn_coo = get_sparse_matrix(trn_f)
        _, trn_cols = trn_coo.shape
        test_coo = get_sparse_matrix(test_f)

        print('Train size:', trn_coo.shape)
        print('Test size:', test_coo.shape)

        print("Considering only the vocab found in train..")
        # this will have same vocab size as the train
        test_csc = test_coo.tocsc()
        test_sub = test_csc[:, :trn_cols].tocoo()
        print('Test_sub size:', test_sub.shape)

        os.system("head -" + str(trn_cols) + " " + data_d +
                  "/vocabulary.txt > " + data_d + "/vocab_train.txt")

        sio.mmwrite(data_d + '/train.mtx', trn_coo.T)
        sio.mmwrite(data_d + '/test_orig.mtx', test_coo.T)
        sio.mmwrite(data_d + '/test.mtx', test_sub.T)

    else:
        print("20NewsGroup train and test data is ready in", data_d)

    if ARGS.test_batch_size:

        train_loader = DataLoader(TwentyNewsDataset('train'),
                                  shuffle=True, batch_size=4000)
        for batch_idx, (data, labels) in enumerate(train_loader):
            print("Train batch:", batch_idx, data.size(), labels.size())

        test_loader = DataLoader(TwentyNewsDataset('test'),
                                 shuffle=True, batch_size=4000)
        for batch_idx, (data, labels) in enumerate(test_loader):
            print("Test batch:", batch_idx, data.size(), labels.size())


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("-t", "--test_batch_size", action='store_true',
                        help='test batch size')
    ARGS = PARSER.parse_args()
    main()
