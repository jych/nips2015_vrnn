# -*- coding: utf 8 -*-
from __future__ import division
import ipdb
import os
import numpy as np
import tables
import numbers
import fnmatch
import scipy.signal

from cle.cle.utils import segment_axis


class _blizzardEArray(tables.EArray):
    pass


def fetch_blizzard(data_path, shuffle=0, sz=32000, file_name="full_blizzard.h5"):

    hdf5_path = os.path.join(data_path, file_name)

    if not os.path.exists(hdf5_path):
        data_matches = []

        for root, dir_names, file_names in os.walk(data_path):
            for filename in fnmatch.filter(file_names, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))

        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))

            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)

                if shuffle:
                    rnd_idx = np.random.permutation(len(d))
                    d = d[rnd_idx]

                for n, di in enumerate(d):
                    print("Processing line %i of %i" % (n+1, len(d)))

                    if len(di.shape) > 1:
                        di = di[:, 0]

                    e = [r for r in range(0, len(di), sz)]
                    e.append(None)
                    starts = e[:-1]
                    stops = e[1:]
                    endpoints = zip(starts, stops)

                    for i, j in endpoints:
                        di_new = di[i:j]

                        # zero pad
                        if len(di_new) < sz:
                            di_large = np.zeros((sz,), dtype='int16')
                            di_large[:len(di_new)] = di_new
                            di_new = di_large

                        data.append(di_new[None])

        hdf5_file.close()

    hdf5_file = tables.openFile(hdf5_path, mode='r')

    return hdf5_file.root.data


def fetch_blizzard_tbptt(data_path, sz=8000, batch_size=100, file_name="blizzard_tbptt.h5"):

    hdf5_path = os.path.join(data_path, file_name)

    if not os.path.exists(hdf5_path):
        data_matches = []

        for root, dir_names, file_names in os.walk(data_path):
            for filename in fnmatch.filter(file_names, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))

        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))

            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)
                large_d = d[0]

                for i in xrange(1, len(d)):
                    print("Processing line %i of %i" % (i+1, len(d)))
                    di = d[i]

                    if len(di.shape) > 1:
                        di = di[:, 0]

                    large_d = np.concatenate([large_d, di])

                chunk_size = int(np.float(len(large_d) / batch_size))
                seg_d = segment_axis(large_d, chunk_size, 0)
                num_batch = int(np.float((seg_d.shape[-1] - 1)/float(sz)))

                for i in range(num_batch):
                    batch = seg_d[:, i*sz:(i+1)*sz]

                    for j in range(batch_size):
                        data.append(batch[j][None])

        hdf5_file.close()

    hdf5_file = tables.openFile(hdf5_path, mode='r')

    return hdf5_file.root.data


if __name__ == "__main__":
    data_path = '/raid/chungjun/data/blizzard/'
    X = fetch_blizzard(data_path, 1)
    from IPython import embed; embed()
    raise ValueError()
