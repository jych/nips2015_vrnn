import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import Gaussian, KLGaussianGaussian
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
from cle.cle.utils.op import Gaussian_sample
from cle.cle.utils.gpu_op import concatenate

from nips2015_vrnn.datasets.blizzard import Blizzard_tbptt


def main(args):

    trial = int(args['trial'])
    pkl_name = 'vrnn_gauss_%d' % trial
    channel_name = 'valid_nll_upper_bound'

    data_path = args['data_path']
    save_path = args['save_path']

    monitoring_freq = int(args['monitoring_freq'])
    force_saving_freq = int(args['force_saving_freq'])
    reset_freq = int(args['reset_freq'])
    epoch = int(args['epoch'])
    batch_size = int(args['batch_size'])
    m_batch_size = int(args['m_batch_size'])
    x_dim = int(args['x_dim'])
    z_dim = int(args['z_dim'])
    rnn_dim = int(args['rnn_dim'])
    lr = float(args['lr'])
    debug = int(args['debug'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    q_z_dim = 500
    p_z_dim = 500
    p_x_dim = 600
    x2s_dim = 600
    z2s_dim = 500
    target_dim = x_dim

    file_name = 'blizzard_unseg_tbptt'
    normal_params = np.load(data_path + file_name + '_normal.npz')
    X_mean = normal_params['X_mean']
    X_std = normal_params['X_std']

    model = Model()
    train_data = Blizzard_tbptt(name='train',
                                path=data_path,
                                frame_size=x_dim,
                                file_name=file_name,
                                X_mean=X_mean,
                                X_std=X_std)

    valid_data = Blizzard_tbptt(name='valid',
                                path=data_path,
                                frame_size=x_dim,
                                file_name=file_name,
                                X_mean=X_mean,
                                X_std=X_std)

    x = train_data.theano_vars()
    m_x = valid_data.theano_vars()

    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=theano.config.floatX)
        m_x.tag.test_value = np.zeros((15, m_batch_size, x_dim), dtype=theano.config.floatX)

    init_W = InitCell('rand')
    init_U = InitCell('ortho')
    init_b = InitCell('zeros')
    init_b_sig = InitCell('const', mean=0.6)

    x_1 = FullyConnectedLayer(name='x_1',
                              parent=['x_t'],
                              parent_dim=[x_dim],
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

    z_1 = FullyConnectedLayer(name='z_1',
                              parent=['z_t'],
                              parent_dim=[z_dim],
                              nout=z2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    z_2 = FullyConnectedLayer(name='z_2',
                              parent=['z_1'],
                              parent_dim=[z2s_dim],
                              nout=z2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    z_3 = FullyConnectedLayer(name='z_3',
                              parent=['z_2'],
                              parent_dim=[z2s_dim],
                              nout=z2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    z_4 = FullyConnectedLayer(name='z_4',
                              parent=['z_3'],
                              parent_dim=[z2s_dim],
                              nout=z2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    rnn = LSTM(name='rnn',
               parent=['x_4', 'z_4'],
               parent_dim=[x2s_dim, z2s_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

    phi_1 = FullyConnectedLayer(name='phi_1',
                                parent=['x_4', 's_tm1'],
                                parent_dim=[x2s_dim, rnn_dim],
                                nout=q_z_dim,
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

    phi_2 = FullyConnectedLayer(name='phi_2',
                                parent=['phi_1'],
                                parent_dim=[q_z_dim],
                                nout=q_z_dim,
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

    phi_3 = FullyConnectedLayer(name='phi_3',
                                parent=['phi_2'],
                                parent_dim=[q_z_dim],
                                nout=q_z_dim,
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

    phi_4 = FullyConnectedLayer(name='phi_4',
                                parent=['phi_3'],
                                parent_dim=[q_z_dim],
                                nout=q_z_dim,
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

    phi_mu = FullyConnectedLayer(name='phi_mu',
                                 parent=['phi_4'],
                                 parent_dim=[q_z_dim],
                                 nout=z_dim,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)

    phi_sig = FullyConnectedLayer(name='phi_sig',
                                  parent=['phi_4'],
                                  parent_dim=[q_z_dim],
                                  nout=z_dim,
                                  unit='softplus',
                                  cons=1e-4,
                                  init_W=init_W,
                                  init_b=init_b_sig)

    prior_1 = FullyConnectedLayer(name='prior_1',
                                  parent=['s_tm1'],
                                  parent_dim=[rnn_dim],
                                  nout=p_z_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    prior_2 = FullyConnectedLayer(name='prior_2',
                                  parent=['prior_1'],
                                  parent_dim=[p_z_dim],
                                  nout=p_z_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    prior_3 = FullyConnectedLayer(name='prior_3',
                                  parent=['prior_2'],
                                  parent_dim=[p_z_dim],
                                  nout=p_z_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    prior_4 = FullyConnectedLayer(name='prior_4',
                                  parent=['prior_3'],
                                  parent_dim=[p_z_dim],
                                  nout=p_z_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    prior_mu = FullyConnectedLayer(name='prior_mu',
                                   parent=['prior_4'],
                                   parent_dim=[p_z_dim],
                                   nout=z_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    prior_sig = FullyConnectedLayer(name='prior_sig',
                                    parent=['prior_4'],
                                    parent_dim=[p_z_dim],
                                    nout=z_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_1 = FullyConnectedLayer(name='theta_1',
                                  parent=['z_4', 's_tm1'],
                                  parent_dim=[z2s_dim, rnn_dim],
                                  nout=p_x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_2 = FullyConnectedLayer(name='theta_2',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=p_x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_3 = FullyConnectedLayer(name='theta_3',
                                  parent=['theta_2'],
                                  parent_dim=[p_x_dim],
                                  nout=p_x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_4 = FullyConnectedLayer(name='theta_4',
                                  parent=['theta_3'],
                                  parent_dim=[p_x_dim],
                                  nout=p_x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_mu = FullyConnectedLayer(name='theta_mu',
                                   parent=['theta_4'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_sig = FullyConnectedLayer(name='theta_sig',
                                    parent=['theta_4'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    nodes = [rnn,
             x_1, x_2, x_3, x_4,
             z_1, z_2, z_3, z_4,
             phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
             prior_1, prior_2, prior_3, prior_4, prior_mu, prior_sig,
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

        phi_1_t = phi_1.fprop([x_t, s_tm1], params)
        phi_2_t = phi_2.fprop([phi_1_t], params)
        phi_3_t = phi_3.fprop([phi_2_t], params)
        phi_4_t = phi_4.fprop([phi_3_t], params)
        phi_mu_t = phi_mu.fprop([phi_4_t], params)
        phi_sig_t = phi_sig.fprop([phi_4_t], params)

        prior_1_t = prior_1.fprop([s_tm1], params)
        prior_2_t = prior_2.fprop([prior_1_t], params)
        prior_3_t = prior_3.fprop([prior_2_t], params)
        prior_4_t = prior_4.fprop([prior_3_t], params)
        prior_mu_t = prior_mu.fprop([prior_4_t], params)
        prior_sig_t = prior_sig.fprop([prior_4_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)

        z_1_t = z_1.fprop([z_t], params)
        z_2_t = z_2.fprop([z_1_t], params)
        z_3_t = z_3.fprop([z_2_t], params)
        z_4_t = z_4.fprop([z_3_t], params)

        s_t = rnn.fprop([[x_t, z_4_t], [s_tm1]], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_4_t

    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_4_temp), updates) =\
        theano.scan(fn=inner_fn,
                    sequences=[x_4_temp],
                    outputs_info=[s_0, None, None, None, None, None])

    for k, v in updates.iteritems():
        k.default_update = v

    shared_updates[rnn_tm1] = s_temp[-1]
    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)
    theta_1_temp = theta_1.fprop([z_4_temp, s_temp], params)
    theta_2_temp = theta_2.fprop([theta_1_temp], params)
    theta_3_temp = theta_3.fprop([theta_2_temp], params)
    theta_4_temp = theta_4.fprop([theta_3_temp], params)
    theta_mu_temp = theta_mu.fprop([theta_4_temp], params)
    theta_sig_temp = theta_sig.fprop([theta_4_temp], params)

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    recon = Gaussian(x, theta_mu_temp, theta_sig_temp)
    recon_term = recon.mean()
    kl_term = kl_temp.mean()
    nll_upper_bound = recon_term + kl_term
    nll_upper_bound.name = 'nll_upper_bound'

    m_x_1_temp = x_1.fprop([m_x], params)
    m_x_2_temp = x_2.fprop([m_x_1_temp], params)
    m_x_3_temp = x_3.fprop([m_x_2_temp], params)
    m_x_4_temp = x_4.fprop([m_x_3_temp], params)

    m_s_0 = rnn.get_init_state(m_batch_size)

    ((m_s_temp, m_phi_mu_temp, m_phi_sig_temp, m_prior_mu_temp, m_prior_sig_temp, m_z_4_temp), m_updates) =\
        theano.scan(fn=inner_fn,
                    sequences=[m_x_4_temp],
                    outputs_info=[m_s_0, None, None, None, None, None])

    for k, v in m_updates.iteritems():
        k.default_update = v

    m_s_temp = concatenate([m_s_0[None, :, :], m_s_temp[:-1]], axis=0)
    m_theta_1_temp = theta_1.fprop([m_z_4_temp, m_s_temp], params)
    m_theta_2_temp = theta_2.fprop([m_theta_1_temp], params)
    m_theta_3_temp = theta_3.fprop([m_theta_2_temp], params)
    m_theta_4_temp = theta_4.fprop([m_theta_3_temp], params)
    m_theta_mu_temp = theta_mu.fprop([m_theta_4_temp], params)
    m_theta_sig_temp = theta_sig.fprop([m_theta_4_temp], params)

    m_kl_temp = KLGaussianGaussian(m_phi_mu_temp, m_phi_sig_temp, m_prior_mu_temp, m_prior_sig_temp)

    m_recon = Gaussian(m_x, m_theta_mu_temp, m_theta_sig_temp)
    m_recon_term = m_recon.mean()
    m_kl_term = m_kl_temp.mean()
    m_nll_upper_bound = m_recon_term + m_kl_term
    m_nll_upper_bound.name = 'nll_upper_bound'
    m_recon_term.name = 'recon_term'
    m_kl_term.name = 'kl_term'

    max_x = m_x.max()
    mean_x = m_x.mean()
    min_x = m_x.min()
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

    max_phi_sig = m_phi_sig_temp.max()
    mean_phi_sig = m_phi_sig_temp.mean()
    min_phi_sig = m_phi_sig_temp.min()
    max_phi_sig.name = 'max_phi_sig'
    mean_phi_sig.name = 'mean_phi_sig'
    min_phi_sig.name = 'min_phi_sig'

    max_prior_sig = m_prior_sig_temp.max()
    mean_prior_sig = m_prior_sig_temp.mean()
    min_prior_sig = m_prior_sig_temp.min()
    max_prior_sig.name = 'max_prior_sig'
    mean_prior_sig.name = 'mean_prior_sig'
    min_prior_sig.name = 'min_prior_sig'

    model.inputs = [x]
    model.params = params
    model.nodes = nodes
    model.set_updates(shared_updates)

    optimizer = Adam(
        lr=lr
    )

    monitor_fn = theano.function(inputs=[m_x],
                                 outputs=[m_nll_upper_bound, m_recon_term, m_kl_term,
                                          max_phi_sig, mean_phi_sig, min_phi_sig,
                                          max_prior_sig, mean_prior_sig, min_prior_sig,
                                          max_theta_sig, mean_theta_sig, min_theta_sig,
                                          max_x, mean_x, min_x,
                                          max_theta_mu, mean_theta_mu, min_theta_mu],
                                 on_unused_input='ignore')

    extension = [
        GradientClipping(batch_size=batch_size, check_nan=1),
        EpochCount(epoch),
        Monitoring(freq=monitoring_freq,
                   monitor_fn=monitor_fn,
                   ddout=[m_nll_upper_bound, m_recon_term, m_kl_term,
                          max_phi_sig, mean_phi_sig, min_phi_sig,
                          max_prior_sig, mean_prior_sig, min_prior_sig,
                          max_theta_sig, mean_theta_sig, min_theta_sig,
                          max_x, mean_x, min_x,
                          max_theta_mu, mean_theta_mu, min_theta_mu],
                   data=[Iterator(train_data, m_batch_size, start=0, end=112640),
                         Iterator(valid_data, m_batch_size, start=2040064, end=2152704)]),
        Picklize(freq=monitoring_freq, force_save_freq=force_saving_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, force_save_freq=force_saving_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]

    mainloop = Training(
        name=pkl_name,
        data=Iterator(train_data, batch_size, start=0, end=2040064),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[nll_upper_bound],
        extension=extension
    )
    mainloop.run()


if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    main(params)
