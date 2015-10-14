import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import Gaussian
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping,
    WeightNorm
)
from cle.cle.train.opt import Adam
from cle.cle.utils import init_tparams, sharedX
from cle.cle.utils.compat import OrderedDict
from cle.cle.utils.gpu_op import concatenate

from nips2015_vrnn.datasets.blizzard import Blizzard_tbptt


trial = 1
pkl_name = 'rnn_gauss_%d' % trial
channel_name = 'valid_nll'
data_path = '/data/lisatmp3/chungjun/data/blizzard_unseg/'
save_path = '/data/lisatmp/chungjun/nips2015/blizzard/pkl/'

epoch = 4
monitoring_freq = 2000
force_saving_freq = 10000
reset_freq = 4
batch_size = 128
m_batch_size = 1280
frame_size = 200
latent_size = 200
rnn_dim = 4000
x2s_dim = 800
s2x_dim = 800
target_size = frame_size
lr = 3e-4
debug = 1

file_name = 'blizzard_unseg_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']

model = Model()
train_data = Blizzard_tbptt(name='train',
                            path=data_path,
                            frame_size=frame_size,
                            file_name=file_name,
                            X_mean=X_mean,
                            X_std=X_std)

valid_data = Blizzard_tbptt(name='valid',
                            path=data_path,
                            frame_size=frame_size,
                            file_name=file_name,
                            X_mean=X_mean,
                            X_std=X_std)

x = train_data.theano_vars()

if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=theano.config.floatX)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x_1 = FullyConnectedLayer(name='x_1',
                          parent=['x_t'],
                          parent_dim=[frame_size],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

x_2 = FullyConnectedLayer(name='x_2',
                          parent=['x_1'],
                          parent_dim=[x2s_dim],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

x_3 = FullyConnectedLayer(name='x_3',
                          parent=['x_2'],
                          parent_dim=[x2s_dim],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

x_4 = FullyConnectedLayer(name='x_4',
                          parent=['x_3'],
                          parent_dim=[x2s_dim],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

rnn = LSTM(name='rnn',
           parent=['x_4'],
           parent_dim=[x2s_dim],
           nout=rnn_dim,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)

theta_1 = FullyConnectedLayer(name='theta_1',
                              parent=['s_tm1'],
                              parent_dim=[rnn_dim],
                              nout=s2x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_2 = FullyConnectedLayer(name='theta_2',
                              parent=['theta_1'],
                              parent_dim=[s2x_dim],
                              nout=s2x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_3 = FullyConnectedLayer(name='theta_3',
                              parent=['theta_2'],
                              parent_dim=[s2x_dim],
                              nout=s2x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_4 = FullyConnectedLayer(name='theta_4',
                              parent=['theta_3'],
                              parent_dim=[s2x_dim],
                              nout=s2x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_mu = FullyConnectedLayer(name='theta_mu',
                               parent=['theta_4'],
                               parent_dim=[s2x_dim],
                               nout=target_size,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

theta_sig = FullyConnectedLayer(name='theta_sig',
                                parent=['theta_4'],
                                parent_dim=[s2x_dim],
                                nout=target_size,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

nodes = [rnn,
         x_1, x_2, x_3, x_4,
         theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig]

params = OrderedDict()
for node in nodes:
    if node.initialize() is not None:
        params.update(node.initialize())
params = init_tparams(params)

step_count = sharedX(0, name='step_count')
last_rnn = np.zeros((batch_size, rnn_dim*2), dtype=theano.config.floatX)
rnn_tm1 = sharedX(last_rnn, name='rnn_tm1')
shared_updates = OrderedDict()
shared_updates[step_count] = step_count + 1

s_0 = T.switch(T.eq(T.mod(step_count, reset_freq), 0),
               rnn.get_init_state(batch_size), rnn_tm1)

x_1_temp = x_1.fprop([x], params)
x_2_temp = x_2.fprop([x_1_temp], params)
x_3_temp = x_3.fprop([x_2_temp], params)
x_4_temp = x_4.fprop([x_3_temp], params)


def inner_fn(x_t, s_tm1):

    s_t = rnn.fprop([[x_t], [s_tm1]], params)

    return s_t

(s_temp, updates) = theano.scan(fn=inner_fn,
                                sequences=[x_4_temp],
                                outputs_info=[s_0])

for k, v in updates.iteritems():
    k.default_update = v

shared_updates[rnn_tm1] = s_temp[-1]
s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)
theta_1_temp = theta_1.fprop([s_temp], params)
theta_2_temp = theta_2.fprop([theta_1_temp], params)
theta_3_temp = theta_3.fprop([theta_2_temp], params)
theta_4_temp = theta_4.fprop([theta_3_temp], params)
theta_mu_temp = theta_mu.fprop([theta_4_temp], params)
theta_sig_temp = theta_sig.fprop([theta_4_temp], params)

recon = Gaussian(x, theta_mu_temp, theta_sig_temp)
recon_term = recon.mean()
recon_term.name = 'nll'

m_s_0 = rnn.get_init_state(m_batch_size)

(m_s_temp, m_updates) = theano.scan(fn=inner_fn,
                                    sequences=[x_4_temp],
                                    outputs_info=[m_s_0])

for k, v in m_updates.iteritems():
    k.default_update = v

m_s_temp = concatenate([m_s_0[None, :, :], m_s_temp[:-1]], axis=0)
m_theta_1_temp = theta_1.fprop([m_s_temp], params)
m_theta_2_temp = theta_2.fprop([m_theta_1_temp], params)
m_theta_3_temp = theta_3.fprop([m_theta_2_temp], params)
m_theta_4_temp = theta_4.fprop([m_theta_3_temp], params)
m_theta_mu_temp = theta_mu.fprop([m_theta_4_temp], params)
m_theta_sig_temp = theta_sig.fprop([m_theta_4_temp], params)

m_recon = Gaussian(x, m_theta_mu_temp, m_theta_sig_temp)
m_recon_term = m_recon.mean()
m_recon_term.name = 'nll'

max_x = x.max()
mean_x = x.mean()
min_x = x.min()
max_x.name = 'max_x'
mean_x.name = 'mean_x'
min_x.name = 'min_x'

max_theta_mu = m_theta_mu_temp.max()
mean_theta_mu = m_theta_mu_temp.mean()
min_theta_mu = m_theta_mu_temp.min()
max_theta_mu.name = 'max_theta_mu'
mean_theta_mu.name = 'mean_theta_mu'
min_theta_mu.name = 'min_theta_mu'

max_theta_sig = m_theta_sig_temp.max()
mean_theta_sig = m_theta_sig_temp.mean()
min_theta_sig = m_theta_sig_temp.min()
max_theta_sig.name = 'max_theta_sig'
mean_theta_sig.name = 'mean_theta_sig'
min_theta_sig.name = 'min_theta_sig'

model.inputs = [x]
model.params = params
model.nodes = nodes
model.set_updates(shared_updates)

optimizer = Adam(
    lr=lr
)

monitor_fn = theano.function(inputs=[x],
                             outputs=[m_recon_term,
                                      max_theta_sig, mean_theta_sig, min_theta_sig,
                                      max_x, mean_x, min_x,
                                      max_theta_mu, mean_theta_mu, min_theta_mu],
                             on_unused_input='ignore')

extension = [
    GradientClipping(batch_size=batch_size, check_nan=1),
    EpochCount(epoch),
    Monitoring(freq=monitoring_freq,
               monitor_fn=monitor_fn,
               ddout=[m_recon_term,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu],
               data=[Iterator(train_data, m_batch_size, start=0, end=112640),
                     Iterator(valid_data, m_batch_size, start=2040064, end=2152704)]), #112640 is 5%
    Picklize(freq=monitoring_freq, force_save_freq=force_saving_freq, path=save_path),
    EarlyStopping(freq=monitoring_freq, force_save_freq=force_saving_freq, path=save_path, channel=channel_name),
    WeightNorm()
]

mainloop = Training(
    name=pkl_name,
    data=Iterator(train_data, batch_size, start=0, end=2040064),
    model=model,
    optimizer=optimizer,
    cost=recon_term,
    outputs=[recon_term],
    extension=extension
)
mainloop.run()
