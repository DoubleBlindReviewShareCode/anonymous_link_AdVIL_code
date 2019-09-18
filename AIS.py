'''
Estimate the likelihood using annealed importance sampling, see the paper:
Salakhutdinov, R. and Murray, I. (2008), "On the Quantitative Analysis of Deep Belief Networks.
'''
import numpy as np
import tensorflow as tf
import utils
from RBM import RBM, BRRBM, MIXRBM
from DBM import DBM, BRDBM, MIXDBM
from SEMIBM import SEMIBM, BRSEMIBM, MIXSEMIBM
from GRBM import GRBM, BRGRBM, MIXGRBM


def estimate_log_partition_function_semi(data_train, semibm, num_ais_samples=100, beta_weights=None):
    brsemibm = BRSEMIBM(semibm.vis_dim, semibm.hid_dim, data_train)
    if beta_weights is None:
        beta_weights = np.concatenate(
            (np.linspace(
                0, 0.5, 1000, endpoint=False, dtype=np.float32), np.linspace(
                0.5, 0.9, 4000, endpoint=False, dtype=np.float32), np.linspace(
                0.9, 1.0, 10000, endpoint=False, dtype=np.float32)), axis=0)
        # beta_weights = np.linspace(0, 1, 10000, endpoint=False, dtype=np.float32)
    # print 'beta_weights', beta_weights
    # draw (independent) samples from the base model
    v = brsemibm.get_independent_samples_tf(num_samples=num_ais_samples, burn_in_steps=1)

    # compute importance weights
    mxbm = MIXSEMIBM(vis_dim=brsemibm.vis_dim, hid_dim=brsemibm.hid_dim + semibm.hid_dim)

    logweights = -brsemibm.ulogprob_vis(v)
    for beta in beta_weights:
        mxbm.tune(brsemibm, semibm, beta)
        logweights += mxbm.ulogprob_vis(v)
        _, v = mxbm.gibbs_vhv(v)
        logweights -= mxbm.ulogprob_vis(v)

    logweights += semibm.ulogprob_vis(v)

    # compute log partition functions
    ais_logweights = logweights + brsemibm.log_z
    ais_logz = tf.reduce_logsumexp(logweights) - np.log(num_ais_samples) + brsemibm.log_z

    return ais_logweights, ais_logz, v


def likelihood_ais_semi(real_v, semibm, log_z):
    ulogprob = semibm.ulogprob_vis(real_v)
    return ulogprob - log_z


def estimate_log_partition_function(data_train, rbm, num_ais_samples=100, beta_weights=None):
    brrbm = BRRBM(rbm.vis_dim, rbm.hid_dim, data_train)
    if beta_weights is None:
        beta_weights = np.concatenate(
            (np.linspace(
                0, 0.5, 1000, endpoint=False, dtype=np.float32), np.linspace(
                0.5, 0.9, 4000, endpoint=False, dtype=np.float32), np.linspace(
                0.9, 1.0, 10000, endpoint=False, dtype=np.float32)), axis=0)
        # beta_weights = np.linspace(0, 1, 10000, endpoint=False, dtype=np.float32)
    # print 'beta_weights', beta_weights
    # draw (independent) samples from the base model
    v = brrbm.get_independent_samples_tf(num_samples=num_ais_samples, burn_in_steps=1)

    # compute importance weights
    mxbm = MIXRBM(vis_dim=brrbm.vis_dim, hid_dim=brrbm.hid_dim + rbm.hid_dim)

    logweights = -brrbm.ulogprob_vis(v)
    for beta in beta_weights:
        mxbm.tune(brrbm, rbm, beta)
        logweights += mxbm.ulogprob_vis(v)
        _, v = mxbm.gibbs_vhv(v)
        logweights -= mxbm.ulogprob_vis(v)

    logweights += rbm.ulogprob_vis(v)

    # compute log partition functions
    ais_logweights = logweights + brrbm.log_z
    ais_logz = tf.reduce_logsumexp(logweights) - np.log(num_ais_samples) + brrbm.log_z

    return ais_logweights, ais_logz, v


def likelihood_ais(real_v, rbm, log_z):
    ulogprob = rbm.ulogprob_vis(real_v)
    return ulogprob - log_z


def estimate_log_partition_function_gaussian(data_train, rbm, num_ais_samples=100, beta_weights=None):
    brrbm = BRGRBM(rbm.vis_dim, rbm.hid_dim, rbm.sigma, data_train)
    if beta_weights is None:
        beta_weights = np.concatenate(
            (np.linspace(
                0, 0.5, 1000, endpoint=False, dtype=np.float32), np.linspace(
                0.5, 0.9, 4000, endpoint=False, dtype=np.float32), np.linspace(
                0.9, 1.0, 10000, endpoint=False, dtype=np.float32)), axis=0)
        # beta_weights = np.linspace(0, 1, 10000, endpoint=False, dtype=np.float32)
    # print 'beta_weights', beta_weights
    # draw (independent) samples from the base model
    v = brrbm.get_independent_samples_tf(num_samples=num_ais_samples, burn_in_steps=1)

    # compute importance weights
    mxbm = MIXGRBM(vis_dim=brrbm.vis_dim, hid_dim=brrbm.hid_dim + rbm.hid_dim)

    logweights = -brrbm.ulogprob_vis(v)
    const_sum = 0
    for beta in beta_weights:
        mxbm.tune(brrbm, rbm, beta)
        logweights += mxbm.ulogprob_vis(v)
        # const_sum += mxbm.log_constant(brrbm, rbm, beta)
        _, v = mxbm.gibbs_vhv(v)
        logweights -= mxbm.ulogprob_vis(v)

    logweights += rbm.ulogprob_vis(v)

    # print 'log const sum ', const_sum

    # compute log partition functions
    ais_logweights = logweights + brrbm.log_z
    ais_logz = tf.reduce_logsumexp(logweights) - np.log(num_ais_samples) + brrbm.log_z

    return ais_logweights, ais_logz, v


def likelihood_ais_gaussian(real_v, rbm, log_z):
    ulogprob = rbm.ulogprob_vis(real_v)
    return ulogprob - log_z


def estimate_log_partition_function_dbm(dbm, num_ais_samples=100, beta_weights=None):
    if beta_weights is None:
        beta_weights = np.concatenate(
            (np.linspace(
                0, 0.5, 1000, endpoint=False, dtype=np.float32), np.linspace(
                0.5, 0.9, 4000, endpoint=False, dtype=np.float32), np.linspace(
                0.9, 1.0, 10000, endpoint=False, dtype=np.float32)), axis=0)
        # beta_weights = np.linspace(0, 1, 20000, endpoint=False, dtype=np.float32)
    # print 'beta_weights', beta_weights
    # draw (independent) samples from the base model
    brdbm = BRDBM(vis_dim=dbm.vis_dim, hid_dim1=dbm.hid_dim1, hid_dim2=dbm.hid_dim2)
    h1 = brdbm.get_independent_h1_samples(num_samples=num_ais_samples, burn_in_steps=1)

    # compute importance weights
    mxdbm = MIXDBM(vis_dim=dbm.vis_dim, hid_dim1=dbm.hid_dim1, hid_dim2=dbm.hid_dim2)

    logweights = -brdbm.ulogprob_h1(h1)
    for beta in beta_weights:
        mxdbm.tune(dbm, beta)
        logweights += mxdbm.ulogprob_h1(h1)
        _, h1, _ = mxdbm.gibbs_hvh(h1)
        logweights -= mxdbm.ulogprob_h1(h1)

    logweights += dbm.ulogprob_h1(h1)

    # compute log partition functions
    ais_logweights = logweights + brdbm.log_z
    ais_logz = tf.reduce_logsumexp(logweights) - np.log(num_ais_samples) + brdbm.log_z

    return ais_logweights, ais_logz, h1


def lowerbound_dbm(real_v, dbm, enc, log_z):
    log_enc_h_given_v, real_h1, real_h2 = enc.log_conditional_prob(real_v)
    log_unnormalized_v_h = - dbm.energy(real_h2, real_h1, real_v)
    return log_unnormalized_v_h - log_enc_h_given_v - log_z
