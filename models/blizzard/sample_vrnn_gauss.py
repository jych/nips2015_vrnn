import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano

from cle.cle.utils import unpickle
from cle.cle.utils.op import Gaussian_sample

from scipy.io import wavfile


data_path = '/data/lisatmp3/chungjun/data/blizzard_unseg/'
save_path = '/data/lisatmp/chungjun/nips2015/blizzard/sample/'
pkl_path = '/data/lisatmp/chungjun/nips2015/blizzard/pkl/'

frame_size = 200
# How many samples to generate
batch_size = 10
# How many timesteps to generate
n_steps = 160
debug = 1

pkl_name = 'vrnn_gauss_1_30000updates.pkl'
save_name = 'vrnn_gauss_1_30000updates_sample_'

file_name = 'blizzard_unseg_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']

exp = unpickle(pkl_path + pkl_name)
nodes = exp.model.nodes
names = [node.name for node in nodes]
params = exp.model.params

[rnn,
 x_1, x_2, x_3, x_4,
 z_1, z_2, z_3, z_4,
 phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
 prior_1, prior_2, prior_3, prior_4, prior_mu, prior_sig,
 theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig] = nodes

s_0 = rnn.get_init_state(batch_size)


def inner_fn(s_tm1):

    prior_1_t = prior_1.fprop([s_tm1], params)
    prior_2_t = prior_2.fprop([prior_1_t], params)
    prior_3_t = prior_3.fprop([prior_2_t], params)
    prior_4_t = prior_4.fprop([prior_3_t], params)
    prior_mu_t = prior_mu.fprop([prior_4_t], params)
    prior_sig_t = prior_sig.fprop([prior_4_t], params)

    z_t = Gaussian_sample(prior_mu_t, prior_sig_t)

    z_1_t = z_1.fprop([z_t], params)
    z_2_t = z_2.fprop([z_1_t], params)
    z_3_t = z_3.fprop([z_2_t], params)
    z_4_t = z_4.fprop([z_3_t], params)

    theta_1_t = theta_1.fprop([z_4_t, s_tm1], params)
    theta_2_t = theta_2.fprop([theta_1_t], params)
    theta_3_t = theta_3.fprop([theta_2_t], params)
    theta_4_t = theta_4.fprop([theta_3_t], params)
    theta_mu_t = theta_mu.fprop([theta_4_t], params)
    theta_sig_t = theta_sig.fprop([theta_4_t], params)

    x_t = Gaussian_sample(theta_mu_t, theta_sig_t)

    x_1_t = x_1.fprop([x_t], params)
    x_2_t = x_2.fprop([x_1_t], params)
    x_3_t = x_3.fprop([x_2_t], params)
    x_4_t = x_4.fprop([x_3_t], params)

    s_t = rnn.fprop([[x_4_t, z_4_t], [s_tm1]], params)

    return s_t, x_t, theta_mu_t

((s_t, y, m), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[s_0, None, None],
                n_steps=n_steps)

for k, v in updates.iteritems():
    k.default_update = v

test_fn = theano.function(inputs=[],
                          outputs=[y, m],
                          updates=updates,
                          allow_input_downcast=True,
                          on_unused_input='ignore')

samples = test_fn()[-1]
samples = np.transpose(samples, (1, 0, 2))
samples = samples.reshape(batch_size, -1)
samples = samples * X_std + X_mean

if not os.path.exists(save_path):
    os.mkdir(save_path)

for i, sample in enumerate(samples):
    wave_path = save_path + save_name + str(i) + '.wav'
    wavfile.write(wave_path, 16000, np.int16(sample))
    sample_path = save_path + save_name + str(i) + '.png'
    fig = plt.figure()
    plt.plot(sample)
    plt.savefig(sample_path, bbox_inches='tight', format='png')
