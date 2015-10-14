import ipdb
import os
import numpy as np
import scipy.signal
import theano
import theano.tensor as T

from cle.cle.data import TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple
from cle.cle.utils.op import overlap_sum, complex_to_real, batch_overlap_sum

from nips2015_vrnn.datasets.grep_blizzard_hdf5 import fetch_blizzard, fetch_blizzard_tbptt

from scipy.io import wavfile


class Blizzard(TemporalSeries, SequentialPrepMixin):
    """
    Blizzard dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 X_mean=None,
                 X_std=None,
                 shuffle=0,
                 seq_len=32000,
                 use_window=0,
                 use_spec=0,
                 frame_size=200,
                 overlap=0,
                 file_name="full_blizzard",
                 **kwargs):

        self.X_mean = X_mean
        self.X_std = X_std
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.use_window = use_window
        self.use_spec = use_spec
        self.frame_size = frame_size
        self.file_name = file_name
        self.overlap = overlap

        if self.use_window or self.use_spec:
            if self.use_spec:
                if not is_power2(self.frame_size):
                    raise ValueError("Provide a number which is power of 2,\
                                      for fast speed of DFT.")

            if np.mod(self.frame_size, 2) == 0:
                self.overlap = self.frame_size / 2
            else:
                self.overlap = (self.frame_size - 1) / 2

            self.window = np.maximum(scipy.signal.hann(self.frame_size)[None, :], 1e-4).astype(theano.config.floatX)

        super(Blizzard, self).__init__(**kwargs)

    def load(self, data_path):

        X = fetch_blizzard(data_path, self.shuffle, self.seq_len, self.file_name+'.h5')

        if (self.X_mean is None or self.X_std is None) and not self.use_spec:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 10000
            range_end = np.int(np.ceil(len(X) / float(n_inter)))

            for i in xrange(range_end):
                n_seen += 1
                i_start = i*n_inter
                i_end = min((i+1)*n_inter, len(X))

                if prev_mean is None:
                    prev_mean = X[i_start:i_end].mean()
                    prev_var = 0.
                else:
                    curr_mean = prev_mean +\
                        (X[i_start:i_end] - prev_mean).mean() / n_seen
                    curr_var = prev_var +\
                        ((X[i_start:i_end] - prev_mean) *
                         (X[i_start:i_end] - curr_mean)).mean()
                    prev_mean = curr_mean
                    prev_var = curr_var

                print "[%d / %d]" % (i+1, range_end)

            save_file_name = self.file_name + '_normal.npz'
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            np.savez(data_path + save_file_name, X_mean=self.X_mean, X_std=self.X_std)

        return X

    def theano_vars(self):
        return T.tensor3('x', dtype=theano.config.floatX)

    def test_theano_vars(self):
        return T.matrix('x', dtype=theano.config.floatX)

    def slices(self, start, end):

        batch = np.array(self.data[start:end], dtype=theano.config.floatX)

        if self.use_spec:
            batch = self.apply_fft(batch)
            batch = self.log_magnitude(batch)
            batch = self.concatenate(batch)
        else:
            batch -= self.X_mean
            batch /= self.X_std
            if self.use_window:
                batch = self.apply_window(batch)
            else:
                batch = np.asarray([segment_axis(x, self.frame_size, 0) for x in batch])

        batch = batch.transpose(1, 0, 2)

        return totuple(batch)

    def apply_window(self, batch):

        batch = np.array([self.window * segment_axis(x, self.frame_size,
                                                     self.overlap, end='pad')
                          for x in batch])

        return batch

    def apply_fft(self, batch):

        batch = np.array([self.numpy_rfft(self.window *
                                          segment_axis(x, self.frame_size,
                                                       self.overlap, end='pad'))
                          for x in batch])

        return batch

    def apply_ifft(self, batch):

        batch = np.array([self.numpy_irfft(example) for example in batch])

        return batch

    def log_magnitude(self, batch):

        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0] *
                                        batch_shape[1],
                                        batch_shape[2]))

        # Transform into polar domain (magnitude & phase)
        mag, phase = R2P(batch_reshaped)
        log_mag = np.log10(mag + 1.)

        # Transform back into complex domain (real & imag)
        batch_normalized = P2R(log_mag, phase)

        #batch_normalized = batch_reshaped * log_mag / mag
        new_batch = batch_normalized.reshape((batch_shape[0],
                                              batch_shape[1],
                                              batch_shape[2]))

        return new_batch

    def pow_magnitude(self, batch):

        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0] *
                                        batch_shape[1],
                                        batch_shape[2]))

        # Transform into polar domain (magnitude & phase)
        log_mag, phase = R2P(batch_reshaped)
        mag = 10**log_mag - 1.

        # Transform back into complex domain (real & imag)
        batch_unnormalized = P2R(mag, phase)

        #batch_unnormalized = batch_reshaped * mag / log_mag
        new_batch = batch_unnormalized.reshape((batch_shape[0],
                                                batch_shape[1],
                                                batch_shape[2]))

        return new_batch

    def concatenate(self, batch):

        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0] *
                                        batch_shape[1],
                                        batch_shape[2]))
        batch_concatenated = complex_to_real(batch_reshaped)
        new_batch = batch_concatenated.reshape((batch_shape[0],
                                                batch_shape[1],
                                                batch_concatenated.shape[-1]))
        new_batch = new_batch.astype(theano.config.floatX)

        return new_batch


class Blizzard_tbptt(Blizzard):
    """
    Blizzard dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 batch_size=100,
                 file_name='blizzard_tbptt',
                 range_start=0,
                 range_end=None,
                 **kwargs):

        self.batch_size = batch_size
        self.range_start = range_start
        self.range_end = range_end
        super(Blizzard_tbptt, self).__init__(file_name=file_name, **kwargs)

    def load(self, data_path):

        X = fetch_blizzard_tbptt(data_path, self.seq_len, self.batch_size,
                                 file_name=self.file_name+'.h5')

        if (self.X_mean is None or self.X_std is None) and not self.use_spec:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 10000
            range_start = self.range_start

            if self.range_end is not None:
                range_end = np.int(np.ceil(self.range_end / float(n_inter)))
            else:
                range_end = np.int(np.ceil(len(X) / float(n_inter)))

            for i in xrange(range_start, range_end):
                n_seen += 1
                i_start = i*n_inter
                i_end = min((i+1)*n_inter, len(X))

                if prev_mean is None:
                    prev_mean = X[i_start:i_end].mean()
                    prev_var = 0.
                else:
                    curr_mean = prev_mean +\
                        (X[i_start:i_end] - prev_mean).mean() / n_seen
                    curr_var = prev_var +\
                        ((X[i_start:i_end] - prev_mean) *\
                         (X[i_start:i_end] - curr_mean)).mean()
                    prev_mean = curr_mean
                    prev_var = curr_var

                print "[%d / %d]" % (i+1, range_end)

            save_file_name = self.file_name + '_normal.npz'
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            np.savez(data_path + save_file_name, X_mean=self.X_mean, X_std=self.X_std)

        return X


def P2R(magnitude, phase):
    return magnitude * np.exp(1j*phase)


def R2P(x):
    return np.abs(x), np.angle(x)


def is_power2(num):
    """
    States if a number is a power of two (Author: A.Polino)
    """
    return num != 0 and ((num & (num - 1)) == 0)


if __name__ == "__main__":

    data_path = '/data/lisatmp3/chungjun/data/blizzard_unseg/'
    frame_size = 200
    seq_len = 32000
    test_tbptt = 1
    use_window = 0
    use_spec = 0

    if test_tbptt:
        seq_len = 8000
        file_name = 'blizzard_unseg_tbptt'
        batch_size = 128
        normal_params = np.load(data_path + file_name + '_normal.npz')
        X_mean = normal_params['X_mean']
        X_std = normal_params['X_std']
        range_start = 0
        range_end = 2040000

        blizzard = Blizzard_tbptt(name='train',
                                  path=data_path,
                                  frame_size=frame_size,
                                  seq_len=seq_len,
                                  use_window=use_window,
                                  file_name=file_name,
                                  X_mean=X_mean,
                                  X_std=X_std,
                                  range_start=range_start,
                                  range_end=range_end,
                                  batch_size=batch_size)
    else:
        file_name = 'full_blizzard'
        normal_params = np.load(data_path + file_name + '_normal.npz')
        X_mean = normal_params['X_mean']
        X_std = normal_params['X_std']

        blizzard = Blizzard(name='train',
                            path=data_path,
                            frame_size=frame_size,
                            seq_len=seq_len,
                            use_window=use_window,
                            use_spec=use_spec,
                            file_name=file_name,
                            X_mean=X_mean,
                            X_std=X_std)

    X_mean = blizzard.X_mean
    X_std = blizzard.X_std
    X = blizzard.data[0]
    batch = blizzard.slices(0, 100)
    ipdb.set_trace()
