import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano.tensor as T

from cle.cle.data import TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple

from iamondb_utils import fetch_iamondb


class IAMOnDB(TemporalSeries, SequentialPrepMixin):
    """
    IAMOnDB dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, prep='none', cond=False, X_mean=None, X_std=None,
                 bias=None, **kwargs):

        self.prep = prep
        self.cond = cond
        self.X_mean = X_mean
        self.X_std = X_std
        self.bias = bias

        super(IAMOnDB, self).__init__(**kwargs)

    def load(self, data_path):

        if self.name == "train":
            X, y, _, _ = fetch_iamondb(data_path)
            print("train")
            print(len(X))
            print(len(y))
        elif self.name == "valid":
            _, _, X, y = fetch_iamondb(data_path)
            print("valid")
            print(len(X))
            print(len(y))

        raw_X = X
        raw_X0 = []
        offset = True
        raw_new_X = []

        for item in raw_X:
            if offset:
                raw_X0.append(item[1:, 0])
                raw_new_X.append(item[1:, 1:] - item[:-1, 1:])
            else:
                raw_X0.append(item[:, 0])
                raw_new_X.append(item[:, 1:])

        raw_new_X, self.X_mean, self.X_std = self.global_normalize(raw_new_X, self.X_mean, self.X_std)
        new_x = []

        for n in range(raw_new_X.shape[0]):
            new_x.append(np.concatenate((raw_X0[n][:, None], raw_new_X[n]),
                                        axis=-1).astype('float32'))
        new_x = np.array(new_x)

        if self.prep == 'none':
            X = np.array(raw_X)

        if self.prep == 'normalize':
            X = new_x
            print X[0].shape
        elif self.prep == 'standardize':
            X, self.X_max, self.X_min = self.standardize(raw_X)

        self.labels = [np.array(y)]

        return [X]

    def theano_vars(self):

        if self.cond:
            return [T.ftensor3('x'), T.fmatrix('mask'),
                    T.ftensor3('y'), T.fmatrix('label_mask')]
        else:
            return [T.ftensor3('x'), T.fmatrix('mask')]

    def theano_test_vars(self):
        return [T.ftensor3('y'), T.fmatrix('label_mask')]

    def slices(self, start, end):

        batches = [mat[start:end] for mat in self.data]
        label_batches = [mat[start:end] for mat in self.labels]
        mask = self.create_mask(batches[0].swapaxes(0, 1))
        batches = [self.zero_pad(batch) for batch in batches]
        label_mask = self.create_mask(label_batches[0].swapaxes(0, 1))
        label_batches = [self.zero_pad(batch) for batch in label_batches]

        if self.cond:
            return totuple([batches[0], mask, label_batches[0], label_mask])
        else:
            return totuple([batches[0], mask])

    def generate_index(self, X):

        maxlen = np.array([len(x) for x in X]).max()
        idx = np.arange(maxlen)

        return idx

if __name__ == "__main__":

    data_path = '/data/lisatmp3/iamondb/'
    iamondb = IAMOnDB(name='train',
                      prep='normalize',
                      cond=False,
                      path=data_path)

    batch = iamondb.slices(start=0, end=10826)
    X = iamondb.data[0]
    sub_X = X

    for item in X:
        max_x = np.max(item[:,1])
        max_y = np.max(item[:,2])
        min_x = np.min(item[:,1])
        min_y = np.min(item[:,2])

    print np.max(max_x)
    print np.max(max_y)
    print np.min(min_x)
    print np.min(min_y)
    ipdb.set_trace()
