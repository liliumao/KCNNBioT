#!/usr/bin/env python
from optparse import OptionParser
import sys

import h5py
import numpy.random as npr
import numpy as np
import pandas as pd
import pdb
import dna_io

################################################################################
# seq_hdf5.py
#
# Make an HDF5 file for Torch input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='add_features_file', default=None, help='Table of additional features')
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Align sizes with batch size')
    parser.add_option('-c', dest='counts', default=False, action='store_true', help='Validation and training proportions are given as raw counts [Default: %default]')
    parser.add_option('-e', dest='extend_length', type='int', default=None, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_pct', default=0, type='float', help='Test % [Default: %default]')
    parser.add_option('-v', dest='valid_pct', default=0, type='float', help='Validation % [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide fasta file, targets file, and an output prefix')
    else:
        fasta_file = args[0]
        targets_file = args[1]
        out_file = args[2]

    # seed rng before shuffle
    npr.seed(options.random_seed)

    #################################################################
    # load data
    #################################################################
    seqs, targets = dna_io.load_data_1hot(fasta_file, targets_file, extend_len=options.extend_length, mean_norm=False, whiten=False, permute=False, sort=False)

    # reshape sequences for torch
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))

    # read headers
    headers = []
    for line in open(fasta_file):
        if line[0] == '>':
            headers.append(line[1:].rstrip())
    headers = np.array(headers)

    # read labels
    target_labels = open(targets_file).readline().strip().split('\t')

    # read additional features
    if options.add_features_file:
        df_add = pd.read_table(options.add_features_file, index_col=0)
        df_add = df_add.astype(np.float32, copy=False)

    # permute
    if options.permute:
        order = npr.permutation(seqs.shape[0])
        seqs = seqs[order]
        targets = targets[order]
        headers = headers[order]
        if options.add_features_file:
            df_add = df_add.iloc[order]

    # check proper sum
    if options.counts:
        assert(options.test_pct + options.valid_pct <= seqs.shape[0])
    else:
        assert(options.test_pct + options.valid_pct <= 1.0)

    #################################################################
    # divide data
    #################################################################
    if options.counts:
        test_count = options.test_pct
        valid_count = options.valid_pct
    else:
        test_count = int(0.5 + options.test_pct * seqs.shape[0])
        valid_count = int(0.5 + options.valid_pct * seqs.shape[0])

    train_count = seqs.shape[0] - test_count - valid_count
    train_count = batch_round(train_count, options.batch_size)
    print >> sys.stderr, '%d training sequences ' % train_count

    test_count = batch_round(test_count, options.batch_size)
    print >> sys.stderr, '%d test sequences ' % test_count

    valid_count = batch_round(valid_count, options.batch_size)
    print >> sys.stderr, '%d validation sequences ' % valid_count

    i = 0
    train_seqs, train_targets = seqs[i:i+train_count,:], targets[i:i+train_count,:]
    i += train_count
    valid_seqs, valid_targets = seqs[i:i+valid_count,:], targets[i:i+valid_count,:]
    i += valid_count
    test_seqs, test_targets, test_headers = seqs[i:i+test_count,:], targets[i:i+test_count,:], headers[i:i+test_count]

    if options.add_features_file:
        i = 0
        train_add = df_add.iloc[i:i+train_count]
        i += train_count
        valid_add = df_add.iloc[i:i+valid_count]
        i += valid_count
        test_add = df_add.iloc[i:i+test_count]

    #################################################################
    # construct hdf5 representation
    #################################################################
    # h5f = h5py.File(out_file, 'w')

    # h5f.create_dataset('target_labels', data=target_labels)

    # pdb.set_trace()
    
    # if train_count > 0:
    #     h5f.create_dataset('train_in', data=train_seqs)
    #     h5f.create_dataset('train_out', data=train_targets)

    # if valid_count > 0:
    #     h5f.create_dataset('valid_in', data=valid_seqs)
    #     h5f.create_dataset('valid_out', data=valid_targets)

    # if test_count > 0:
    #     h5f.create_dataset('test_in', data=test_seqs)
    #     h5f.create_dataset('test_out', data=test_targets)
    #     h5f.create_dataset('test_headers', data=test_headers)

    # if options.add_features_file:
    #     h5f.create_dataset('add_labels', data=list(df_add.columns))

    #     if train_count > 0:
    #         h5f.create_dataset('train_add', data=train_add.as_matrix())
    #     if valid_count > 0:
    #         h5f.create_dataset('valid_add', data=valid_add.as_matrix())
    #     if test_count > 0:
    #         h5f.create_dataset('test_add', data=test_add.as_matrix())

    # h5f.close()
    train_seqs = np.transpose(np.squeeze(train_seqs), (0,2,1))
    train_targets = np.transpose(train_targets, (1, 0))
    valid_seqs = np.transpose(np.squeeze(valid_seqs), (0,2,1))
    valid_targets = np.transpose(valid_targets, (1, 0))
    test_seqs = np.transpose(np.squeeze(test_seqs), (0,2,1))
    test_targets = np.transpose(test_targets, (1, 0))
    
    # pdb.set_trace()

    np.savez(out_file, train_in = train_seqs, train_out = train_targets, valid_in = valid_seqs, valid_out = valid_targets, test_in = test_seqs, test_out = test_targets)


def batch_round(count, batch_size):
    if batch_size != None:
        count -= (batch_size % count)
    return count

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()