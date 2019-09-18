'''
Deep Boltzmann Machines
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import sample_from_bernoulli, tf_xavier_init

class DBM():
    def __init__(self, vis_dim, hid_dim1, hid_dim2, w1=None, w2=None, vis_b=None, hid_b1=None, hid_b2=None):
        self.vis_dim = vis_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        if w1 is not None:
            self.w1 = w1
        else:
            self.w1 = tfe.Variable(tf_xavier_init(self.vis_dim, self.hid_dim1, const=4.0), name='dbm.w1')        
        if w2 is not None:
            self.w2 = w2
        else:
            self.w2 = tfe.Variable(tf_xavier_init(self.hid_dim1, self.hid_dim2, const=4.0), name='dbm.w2')
        if hid_b1 is not None:
            self.hid_b1 = hid_b1
        else:
            self.hid_b1 = tfe.Variable(tf.zeros([self.hid_dim1]), dtype=tf.float32, name='dbm.hid_b1')        
        if hid_b2 is not None:
            self.hid_b2 = hid_b2
        else:
            self.hid_b2 = tfe.Variable(tf.zeros([self.hid_dim2]), dtype=tf.float32, name='dbm.hid_b2')
        if vis_b is not None:
            self.vis_b = vis_b
        else:
            self.vis_b = tfe.Variable(tf.zeros([self.vis_dim]), dtype=tf.float32, name='dbm.vis_b')

    # conditional distributions
    def vh2_to_h1(self, v, h2):
        return tf.nn.sigmoid(tf.matmul(h2, tf.transpose(self.w2)) + tf.matmul(v, self.w1) + self.hid_b1)

    def h1_to_v(self, h1):
        return tf.nn.sigmoid(tf.matmul(h1, tf.transpose(self.w1)) + self.vis_b)
    
    def h1_to_h2(self, h1):
        return tf.nn.sigmoid(tf.matmul(h1, self.w2) + self.hid_b2)

    # Gibbs steps
    def gibbs_vhv(self, v, h2):    
        h1 = sample_from_bernoulli(self.vh2_to_h1(v, h2))
        new_v = sample_from_bernoulli(self.h1_to_v(h1))
        new_h2 = sample_from_bernoulli(self.h1_to_h2(h1))
        return new_v, h1, new_h2

    # Gibbs steps
    def _gibbs_vhv(self, v, h2):    
        h1 = sample_from_bernoulli(self.vh2_to_h1(v, h2))
        new_v = self.h1_to_v(h1)
        new_h2 = sample_from_bernoulli(self.h1_to_h2(h1))
        return new_v, h1, new_h2


    def gibbs_hvh(self, h1):    
        v = sample_from_bernoulli(self.h1_to_v(h1))
        h2 = sample_from_bernoulli(self.h1_to_h2(h1))
        new_h1 = sample_from_bernoulli(self.vh2_to_h1(v, h2))
        return v, new_h1, h2

    # marginalization
    def ulogprob_h1(self, h1):
        wx_b1 = tf.matmul(h1, tf.transpose(self.w1)) + self.vis_b
        wx_b2 = tf.matmul(h1, self.w2) + self.hid_b2
        latent_term = tf.reduce_sum(tf.nn.softplus(wx_b1), axis=1) + tf.reduce_sum(tf.nn.softplus(wx_b2), axis=1)
        hbias_term = tf.einsum('ij,j->i', h1, self.hid_b1)
        return latent_term + hbias_term

    def ulogprob_vis(self, v):
        assert(self.hid_dim1 <= 20)
        h_all = np.arange(2**self.hid_dim1, dtype=np.int32)
        h_all = ((h_all.reshape(-1,1) & (2**np.arange(self.hid_dim1))) != 0).astype(np.float32)
        h_all = tf.constant(h_all[:,::-1], dtype=tf.float32)

        h1_bias = tf.einsum('ij,j->i', h_all, self.hid_b1)
        # print h1_bias.numpy().shape
        h1_weight = tf.matmul(tf.matmul(v, self.w1), tf.transpose(h_all))
        # print h1_weight.numpy().shape
        wx_b2 = tf.matmul(h_all, self.w2) + self.hid_b2
        h2_term = tf.reduce_sum(tf.nn.softplus(wx_b2), axis=1)
        # print h2_term.numpy().shape
        vbias_term = tf.einsum('ij,j->i', v, self.vis_b)
        # print vbias_term.numpy().shape
        return tf.reduce_logsumexp(tf.expand_dims(h2_term, 0) + tf.expand_dims(h1_bias, 0) + h1_weight, axis=1) + vbias_term

    # log partiation function
    def log_z_summing_h1(self):
        assert(self.hid_dim1 <= 20)
        h_all = np.arange(2**self.hid_dim1, dtype=np.int32)
        h_all = ((h_all.reshape(-1,1) & (2**np.arange(self.hid_dim1))) != 0).astype(np.float32)
        h_all = tf.constant(h_all[:,::-1], dtype=tf.float32)
        log_p_h = self.ulogprob_h1(h_all)
        log_z = tf.reduce_logsumexp(log_p_h, axis=0)
        return log_z

    # likelihood
    def logprob_vis(self, v, log_z):
        return self.ulogprob_vis(v) - log_z

    # energy function
    def energy(self, h2, h1, v):
        hbias_term1 = tf.einsum('ij,j->i', h1, self.hid_b1)
        vbias_term = tf.einsum('ij,j->i', v, self.vis_b)
        hbias_term2 = tf.einsum('ij,j->i', h2, self.hid_b2)
        weight_term1 = tf.reduce_sum(tf.matmul(v, self.w1)*h1, axis=1)
        weight_term2 = tf.reduce_sum(tf.matmul(h1, self.w2)*h2, axis=1)
        return - (hbias_term1 + vbias_term + hbias_term2 + weight_term1 + weight_term2)
         
    # get samples
    def get_independent_samples(self, num_samples, burn_in_steps=100000):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        h2 = tf.zeros([num_samples, self.hid_dim2], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            v, _, h2 = self.gibbs_vhv(v, h2)
        return v.numpy()

    def get_independent_means(self, num_samples, burn_in_steps=100000):
        v = tf.zeros([num_samples, self.vis_dim], dtype=tf.float32)
        h2 = tf.zeros([num_samples, self.hid_dim2], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            v, _, h2 = self.gibbs_vhv(v, h2)
        return v.numpy()

    def get_samples_single_chain(self, num_samples, adjacent_samples=10, steps_between_samples=1000, burn_in_steps=100000):
        assert num_samples % adjacent_samples == 0
        v = tf.zeros([1, self.vis_dim], dtype=tf.float32)
        h2 = tf.zeros([1, self.hid_dim2], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            v, _, h2 = self.gibbs_vhv(v, h2)
        sample_list = []
        for i in xrange(num_samples/adjacent_samples):
            for j in xrange(adjacent_samples):
                v, _, h2 = self.gibbs_vhv(v, h2)
                sample_list.append(v.numpy())
            for i in xrange(steps_between_samples):
                v, _, h2 = self.gibbs_vhv(v, h2)
        return np.vstack(sample_list)

    def params(self):
        return (self.hid_b1, self.hid_b2, self.vis_b, self.w1, self.w2)

    def approximate_gibbs_hvh(self, h1, enc):
        v = sample_from_bernoulli(self.h1_to_v(h1))
        new_h1 = enc.get_hard_h1(v)
        return v, new_h1

    # contrastive free energy: see M.Welling 2005
    def cfe_step(self, v, enc, train_mc_steps):
        h1 = enc.get_hard_h1(v)
        h1_list = [
           h1,
        ]
        v_list = []
        for i in xrange(train_mc_steps):
            new_v, new_h1 = self.approximate_gibbs_hvh(h1_list[-1], enc)
            v_list.append(new_v)
            h1_list.append(new_h1)
        chain_end = tf.stop_gradient(v_list[-1])
        return chain_end

# base rate DBM for AIS: uniform distribution
class BRDBM(DBM):
    def __init__(self, vis_dim, hid_dim1, hid_dim2):
        self.vis_dim = vis_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.w1 = tfe.Variable(tf.zeros([vis_dim, hid_dim1]), dtype=tf.float32)
        self.w2 = tfe.Variable(tf.zeros([hid_dim1, hid_dim2]), dtype=tf.float32)
        self.hid_b1 = tfe.Variable(tf.zeros([self.hid_dim1]), dtype=tf.float32)
        self.hid_b2 = tfe.Variable(tf.zeros([self.hid_dim2]), dtype=tf.float32)
        self.vis_b = tfe.Variable(tf.zeros([self.vis_dim]), dtype=tf.float32)
        self.log_z = (self.vis_dim+self.hid_dim1+self.hid_dim2)*np.log(2.)

    # get tf samples
    def get_independent_h1_samples(self, num_samples, burn_in_steps=100):
        h1 = tf.zeros([num_samples, self.hid_dim1], dtype=tf.float32)
        for i in xrange(burn_in_steps):
            _, h1, _ = self.gibbs_hvh(h1)
        return h1

# Mix DBM for AIS
class MIXDBM(DBM):
    def tune(self, dbm, weight):
        self.w1 = weight * dbm.w1
        self.w2 = weight * dbm.w2
        self.hid_b1 = weight * dbm.hid_b1
        self.hid_b2 = weight * dbm.hid_b2
        self.vis_b = weight * dbm.vis_b