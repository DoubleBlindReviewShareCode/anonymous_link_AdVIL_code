'''
Gaussian-Bernoulli Restricted Boltzmann Machines
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import sample_from_bernoulli, sample_from_gaussian, tf_xavier_init


class GRBM():

    def __init__(self, vis_dim, hid_dim, sigma=1., w=None, vis_b=None, hid_b=None):
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim
        self.sigma = sigma
        if w is not None:
            self.w = w
        else:
            self.w = tfe.Variable(tf_xavier_init(self.vis_dim, self.hid_dim, const=4.0), name='rbm.w')
        if hid_b is not None:
            self.hid_b = hid_b
        else:
            self.hid_b = tfe.Variable(tf.zeros([self.hid_dim]), dtype=tf.float32, name='rbm.hid_b')
        if vis_b is not None:
            self.vis_b = vis_b
        else:
            self.vis_b = tfe.Variable(tf.zeros([self.vis_dim]), dtype=tf.float32, name='rbm.vis_b')

    # conditional distributions
    def vis2hid(self, v):
        return tf.nn.sigmoid(tf.matmul(v, self.w) / self.sigma + self.hid_b)

    def hid2vis(self, h):
        return self.sigma * tf.matmul(h, tf.transpose(self.w)) + self.vis_b

    # Gibbs steps
    def gibbs_vhv(self, v_0):
        h_1 = sample_from_bernoulli(self.vis2hid(v_0))
        v_1 = sample_from_gaussian(self.hid2vis(h_1), 2 * tf.log(self.sigma))
        v_1 = tf.stop_gradient(v_1)
        return h_1, v_1

    def gibbs_hvh(self, h_0):
        v_1 = sample_from_gaussian(self.hid2vis(h_0), 2 * tf.log(self.sigma))
        v_1 = tf.stop_gradient(v_1)
        h_1 = sample_from_bernoulli(self.vis2hid(v_1))
        return v_1, h_1

    # marginalization
    def ulogprob_vis(self, v):
        wx_b = tf.matmul(v, self.w) / self.sigma + self.hid_b
        vbias_term = - tf.reduce_sum(tf.square(v - self.vis_b), axis=1) / (2 * (self.sigma**2))
        hidden_term = tf.reduce_sum(tf.nn.softplus(wx_b), axis=1)
        return hidden_term + vbias_term

    def ulogprob_hid(self, h):
        wh = tf.matmul(h, tf.transpose(self.w))
        square_term = .5 * tf.reduce_sum(tf.square(wh), axis=1) + tf.einsum('ij,j->i', wh, self.vis_b) / self.sigma
        hbias_term = tf.einsum('ij,j->i', h, self.hid_b)
        log_z_conditoinal = .5 * self.vis_dim * (tf.log(2 * np.pi * (self.sigma**2)))
        return square_term + hbias_term + log_z_conditoinal

    # log partiation function
    def log_z_summing_h(self):
        assert(self.hid_dim <= 20)
        h_all = np.arange(2**self.hid_dim, dtype=np.int32)
        h_all = ((h_all.reshape(-1, 1) & (2**np.arange(self.hid_dim))) != 0).astype(np.float32)
        h_all = tf.constant(h_all[:, ::-1], dtype=tf.float32)
        log_p_h = self.ulogprob_hid(h_all)
        log_z = tf.reduce_logsumexp(log_p_h, axis=0)
        return log_z

    # def log_z_summing_v(self):
    #     assert(self.vis_dim <= 20)
    #     v_all = np.arange(2**self.vis_dim, dtype=np.int32)
    #     v_all = ((v_all.reshape(-1, 1) & (2**np.arange(self.vis_dim))) != 0).astype(np.float32)
    #     v_all = tf.constant(v_all[:, ::-1], dtype=tf.float32)
    #     log_p_v = self.ulogprob_vis(v_all)
    #     log_z = tf.reduce_logsumexp(log_p_v, axis=0)
    #     return log_z

    # likelihood
    def logprob_vis(self, v, log_z):
        return self.ulogprob_vis(v) - log_z

    def logprob_hid(self, h, log_z):
        return self.ulogprob_hid(h) - log_z

    # energy function
    def energy(self, h, v):
        hbias_term = tf.einsum('ij,j->i', h, self.hid_b)
        vbias_term = tf.reduce_sum(tf.square(v - self.vis_b), axis=1) / (2 * (self.sigma**2))
        weight_term = tf.reduce_sum(tf.matmul(v, self.w) * h, axis=1) / self.sigma
        return vbias_term - hbias_term - weight_term

    # free energy
    def free_energy(self, v):
        return -self.ulogprob_vis(v)

    # free energy for debug
    # def _debug_free_energy(self, v):
    #     assert (self.hid_dim <= 20)
    #     assert (v.numpy().shape == (1, self.vis_dim))

    #     h_all = np.arange(2**self.hid_dim, dtype=np.int32)
    #     h_all = ((h_all.reshape(-1, 1) & (2**np.arange(self.hid_dim))) != 0).astype(np.float32)
    #     h_all = tf.constant(h_all[:, ::-1], dtype=tf.float32)
    #     v_dup = tf.tile(v, [2**self.hid_dim, 1])
    #     return -tf.reduce_logsumexp(-self.energy(h_all, v), axis=0)

    # # get samples
    # def get_h_from_v(self, v, burn_in_steps=100):
    #     for i in xrange(burn_in_steps):
    #         h, v = self.gibbs_vhv(v)
    #     return h.numpy()

    # def get_h(self, num_samples, burn_in_steps=1000, random=True):
    #     v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
    #     if random:
    #         v = sample_from_bernoulli(v + 0.5)  # data average
    #     for i in xrange(burn_in_steps):
    #         h, v = self.gibbs_vhv(v)
    #     return h.numpy()

    def get_independent_samples(self, num_samples, burn_in_steps=100000, random=True, initial_v=None):
        if initial_v is not None:
            v = initial_v
        else:
            v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32) + self.vis_b
            if random:
                v = sample_from_gaussian(v, 2. * tf.log(self.sigma))
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        return v.numpy()

    def get_independent_means(self, num_samples, burn_in_steps=100000, random=True):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        if random:
            v = sample_from_gaussian(v, 2. * tf.log(self.sigma)) + self.vis_b
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        h_1 = sample_from_bernoulli(self.vis2hid(v))
        v_1 = self.hid2vis(h_1)
        return v_1.numpy()

    def get_samples_single_chain(self, num_samples, adjacent_samples=10, steps_between_samples=1000, burn_in_steps=100000, random=True):
        assert num_samples % adjacent_samples == 0
        v = tf.zeros([1, self.vis_dim], dtype=tf.float32) + self.vis_b
        if random:
            v = sample_from_gaussian(v, 2. * tf.log(self.sigma))
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        sample_list = []
        for i in xrange(num_samples / adjacent_samples):
            for j in xrange(adjacent_samples):
                _, v = self.gibbs_vhv(v)
                sample_list.append(v.numpy())
            for i in xrange(steps_between_samples):
                _, v = self.gibbs_vhv(v)
        return np.vstack(sample_list)

    # for constrastive divergence training
    def cd_step(self, v, train_mc_steps):
        h = sample_from_bernoulli(self.vis2hid(v))
        h_list = [h, ]
        v_list = []
        for i in xrange(train_mc_steps):
            new_v, new_h = self.gibbs_hvh(h_list[-1])
            v_list.append(new_v)
            h_list.append(new_h)
        chain_end = tf.stop_gradient(v_list[-1])
        return chain_end

    def pcd_step(self, v, train_mc_steps, persistent):
        h_list = [persistent, ]
        v_list = []
        for i in xrange(train_mc_steps):
            new_v, new_h = self.gibbs_hvh(h_list[-1])
            v_list.append(new_v)
            h_list.append(new_h)
        chain_end = tf.stop_gradient(v_list[-1])
        return chain_end, tf.stop_gradient(h_list[-1])

    def cd_loss(self, v_0, v_n):
        return tf.reduce_mean(self.free_energy(v_0), axis=0) - tf.reduce_mean(self.free_energy(v_n), axis=0)

    # reconstruction
    def reconstruction_error(self, v_0):
        h_1 = sample_from_bernoulli(self.vis2hid(v_0))
        v_1_logits = tf.matmul(h_1, tf.transpose(self.w)) + self.vis_b
        return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=v_0, logits=v_1_logits), axis=1), axis=0)

    def params(self):
        return (self.hid_b, self.vis_b, self.w)

# base rate RBM for AIS


class BRGRBM(GRBM):
    def __init__(self, vis_dim, hid_dim, sigma, data):
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim
        self.sigma = sigma
        self.w = tfe.Variable(tf.zeros([vis_dim, hid_dim]), dtype=tf.float32)
        self.hid_b = tfe.Variable(tf.zeros([self.hid_dim]), dtype=tf.float32)
        # MLE for the value of vis_b
        sample_mean = tf.reduce_mean(data, axis=0)
        # Smooth to make sure p(v) > 0 for every v
        # sample_mean = tf.clip_by_value(sample_mean, 1e-5, 1 - 1e-5)
        self.vis_b = sample_mean
        self.log_z = .5 * vis_dim * (tf.log(2 * np.pi * (self.sigma**2))) + self.hid_dim * np.log(2.)

    # get tf samples
    def get_independent_samples_tf(self, num_samples, burn_in_steps=100):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        return v

# Mix RBM for AIS


class MIXGRBM(GRBM):
    def tune(self, brrbm, rbm, weight):
        # adjust parameters of the mixed RBM
        n = brrbm.hid_dim
        assert (brrbm.sigma == rbm.sigma)
        # self.vis_b = (1. - weight) * brrbm.vis_b + weight * rbm.vis_b
        # self.hid_b = tf.concat([(1. - weight) * brrbm.hid_b, weight * rbm.hid_b], axis=0)
        # self.w = tf.concat([(1. - weight) * brrbm.w, weight * rbm.w], axis=1)

        self.vis_b = (1. - weight) * brrbm.vis_b + weight * rbm.vis_b
        self.hid_b = tf.concat([(1. - weight) * brrbm.hid_b, weight * rbm.hid_b], axis=0)
        self.w = tf.concat([(1. - weight) * brrbm.w, weight * rbm.w], axis=1)

    def log_constant(self, brrbm, rbm, weight):
        return tf.reduce_sum(tf.square(brrbm.vis_b - rbm.vis_b)) * weight * (1 - weight) / (2 * self.sigma**2)
