'''
SEMI-Restricted Boltzmann Machines
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import sample_from_bernoulli, tf_xavier_init

class SEMIBM():
    def __init__(self, vis_dim, hid_dim, w=None, l=None, vis_b=None, hid_b=None):
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim
        if w is not None:
            self.w = w
        else:
            self.w = tfe.Variable(tf_xavier_init(self.vis_dim, self.hid_dim, const=4.0), name='semibm.w')        
        if l is not None:
            self.l = l
        else:
            self.l = tfe.Variable(tf_xavier_init(self.vis_dim, self.vis_dim, const=4.0), name='semibm.l')
        if hid_b is not None:
            self.hid_b = hid_b
        else:
            self.hid_b = tfe.Variable(tf.zeros([self.hid_dim]), dtype=tf.float32, name='semibm.hid_b')
        if vis_b is not None:
            self.vis_b = vis_b
        else:
            self.vis_b = tfe.Variable(tf.zeros([self.vis_dim]), dtype=tf.float32, name='semibm.vis_b')

    # conditional distributions
    def vis2hid(self, v):
        return tf.nn.sigmoid(tf.matmul(v, self.w) + self.hid_b)

    def hid2vis(self, h, v):
        uncorr_term = tf.matmul(h, tf.transpose(self.w)) + self.vis_b
        tmp_v = v
        mu, sample = [],[]
        
        def loop_body(i, tmp_v):
            corr_term = tf.reduce_sum(tf.matmul(tmp_v, self.l)*tmp_v, axis=1)
            corr_term -= tf.reduce_sum(tf.expand_dims(tmp_v[:, i], axis=1) * tf.expand_dims(self.l[:, i], axis=0), axis=1)
            corr_term -= tf.reduce_sum(tf.expand_dims(tmp_v[:, i], axis=1) * tf.expand_dims(self.l[i, :], axis=0), axis=1)
            corr_term += tmp_v[:, i] * tmp_v[:, i] * self.l[i, i]
            mu_i = tf.sigmoid(uncorr_term[:, i] + corr_term)
            sample_i = sample_from_bernoulli(mu_i)
            return tf.expand_dims(mu_i, axis=1), tf.expand_dims(sample_i, axis=1)

        for i in range(self.vis_dim):
            mu_i, sample_i = loop_body(i, tmp_v)
            tmp_v_list = []
            if i > 0:
                tmp_v_list.append(tmp_v[:,:i])
            tmp_v_list.append(sample_i)
            if i < self.vis_dim - 1:
                tmp_v_list.append(tmp_v[:,i+1:])
            tmp_v = tf.concat(tmp_v_list, axis=1)
            mu.append(mu_i)
            sample.append(sample_i)
        return tf.concat(mu, axis=1), tf.concat(sample, axis=1)
    
    # Gibbs steps
    def gibbs_vhv(self, v_0):    
        h_1 = sample_from_bernoulli(self.vis2hid(v_0))
        v_1 = self.hid2vis(h_1, v_0)[1]
        return h_1, v_1

    def gibbs_hvh(self, h_0, v_0):    
        v_1 = self.hid2vis(h_0, v_0)[1]
        h_1 = sample_from_bernoulli(self.vis2hid(v_1))
        return v_1, h_1

    # marginalization
    def ulogprob_vis(self, v):
        wx_b = tf.matmul(v, self.w) + self.hid_b
        vbias_term = tf.einsum('ij,j->i', v, self.vis_b)
        vbias_term2 = tf.reduce_sum(tf.matmul(v, self.l)*v, axis=1) 
        hidden_term = tf.reduce_sum(tf.nn.softplus(wx_b), axis=1)
        return hidden_term + vbias_term + vbias_term2

    def log_z_summing_v(self):
        assert(self.vis_dim <= 20)
        v_all = np.arange(2**self.vis_dim, dtype=np.int32)
        v_all = ((v_all.reshape(-1,1) & (2**np.arange(self.vis_dim))) != 0).astype(np.float32)
        v_all = tf.constant(v_all[:,::-1], dtype=tf.float32)
        log_p_v = self.ulogprob_vis(v_all)
        log_z = tf.reduce_logsumexp(log_p_v, axis=0)
        return log_z

    def log_z_summing_vh(self):
        assert(self.vis_dim <= 20)
        v_all = np.arange(2**self.vis_dim, dtype=np.int32)
        v_all = ((v_all.reshape(-1,1) & (2**np.arange(self.vis_dim))) != 0).astype(np.float32)
        v_all = tf.constant(v_all[:,::-1], dtype=tf.float32)

        assert(self.hid_dim <= 3)
        h_all = np.arange(2**self.hid_dim, dtype=np.int32)
        h_all = ((h_all.reshape(-1,1) & (2**np.arange(self.hid_dim))) != 0).astype(np.float32)
        h_all = tf.constant(h_all[:,::-1], dtype=tf.float32)
        
        log_z = []
        for h in h_all:
            log_z.append(tf.reduce_logsumexp(- self.energy(tf.expand_dims(h, axis=0), v_all), keep_dims=True))

        log_z = tf.reduce_logsumexp(tf.concat(log_z, axis=0), axis=0)

        return log_z

    # likelihood
    def logprob_vis(self, v, log_z):
        return self.ulogprob_vis(v) - log_z

    # energy function
    def energy(self, h, v):
        hbias_term = tf.einsum('ij,j->i', h, self.hid_b)
        vbias_term = tf.einsum('ij,j->i', v, self.vis_b)
        weight_term = tf.reduce_sum(tf.matmul(v, self.w)*h, axis=1)
        weight_term2 = tf.reduce_sum(tf.matmul(v, self.l)*v, axis=1)
        return - (hbias_term + vbias_term + weight_term + weight_term2)

    # free energy
    def free_energy(self, v):
        return -self.ulogprob_vis(v)
         
    # get samples
    def get_independent_samples(self, num_samples, burn_in_steps=100000):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        return v.numpy()

    def get_independent_means(self, num_samples, burn_in_steps=100000):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        h_1 = sample_from_bernoulli(self.vis2hid(v))
        v_1 = self.hid2vis(h_1)[0]
        return v_1.numpy()    

    def get_samples_single_chain(self, num_samples, adjacent_samples=10, steps_between_samples=1000, burn_in_steps=100000):
        assert num_samples % adjacent_samples == 0
        v = tf.zeros([1, self.vis_dim], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        sample_list = []
        for i in xrange(num_samples/adjacent_samples):
            for j in xrange(adjacent_samples):
                _, v = self.gibbs_vhv(v)
                sample_list.append(v.numpy())
            for i in xrange(steps_between_samples):
                _, v = self.gibbs_vhv(v)
        return np.vstack(sample_list)

    def cd_loss(self, v_0, v_n):
        return tf.reduce_mean(self.free_energy(v_0), axis=0) - tf.reduce_mean(self.free_energy(v_n), axis=0)

    def params(self):
        return (self.hid_b, self.vis_b, self.w, self.l)

# base rate RBM for AIS
class BRSEMIBM(SEMIBM):
    def __init__(self, vis_dim, hid_dim, data):
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim
        self.w = tfe.Variable(tf.zeros([vis_dim, hid_dim]), dtype=tf.float32)
        self.l = tfe.Variable(tf.zeros([vis_dim, vis_dim]), dtype=tf.float32)
        self.hid_b = tfe.Variable(tf.zeros([self.hid_dim]), dtype=tf.float32)
        # MLE for the value of vis_b
        sample_mean = tf.reduce_mean(data, axis=0)
        # Smooth to make sure p(v) > 0 for every v
        sample_mean = tf.clip_by_value(sample_mean, 1e-5, 1-1e-5)
        self.vis_b = -tf.log(1. / sample_mean - 1.)
        self.log_z = tf.reduce_sum(tf.nn.softplus(self.vis_b), axis=0) + self.hid_dim*np.log(2.)

    # get tf samples
    def get_independent_samples_tf(self, num_samples, burn_in_steps=100):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            _, v = self.gibbs_vhv(v)
        return v

# Mix RBM for AIS
class MIXSEMIBM(SEMIBM):
    def tune(self, brsemibm, semibm, weight):
        # adjust parameters of the mixed RBM
        self.vis_b = (1. - weight) * brsemibm.vis_b + weight * semibm.vis_b
        self.hid_b = tf.concat([(1. - weight) * brsemibm.hid_b, weight * semibm.hid_b], axis=0)
        self.w = tf.concat([(1. - weight) * brsemibm.w, weight * semibm.w], axis=1)
        self.l = (1. - weight) * brsemibm.l + weight * semibm.l