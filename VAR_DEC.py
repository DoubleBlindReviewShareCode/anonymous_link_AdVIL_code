from NADE import NADE
'''
Probabilistic decoders as variational approximation for sampling
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import gumbel_sigmoid_sample, gumbel_softmax_sample, sample_from_bernoulli, bernoulli_log_likelihood, tf_xavier_init
from VAR_ENC import ENC_CONT


class SBNPrior():

    def __init__(self,
                 dims,
                 name,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False):
        self.dims = dims
        self.ws, self.bs = [], []

        assert len(self.dims) > 1
        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        for d1, d2, i in zip(dims[:-1], dims[1:], xrange(1, len(self.dims))):
            self.ws.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='dec.' + name + '.w.' + str(i)))
            self.bs.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='dec.' + name + '.b.' + str(i)))

        self.activation_fn = activation_fn
        self.temp = temp
        self.hard = hard

    def inference(self, z):
        h = z
        for w, b, i in zip(self.ws, self.bs, xrange(len(self.bs))):
            h = tf.matmul(h, w) + b
            if i == len(self.ws) - 1:
                h = tf.nn.sigmoid(h)
            else:
                h = self.activation_fn(h)
        return h

    def sample(self, num_samples):
        z = sample_from_bernoulli(
            tf.constant(
                .5, shape=(num_samples, self.dims[0]), dtype=tf.float32))
        h2 = gumbel_sigmoid_sample(self.inference(z), self.temp, self.hard)
        return h2

    def logprobs_all(self, num_samples):
        z = sample_from_bernoulli(
            tf.constant(
                .5, shape=(num_samples, self.dims[0]), dtype=tf.float32))
        mu = self.inference(z)
        h = gumbel_sigmoid_sample(mu, self.temp, self.hard)
        logp_z = bernoulli_log_likelihood(
            z, tf.constant(.5, shape=z.shape, dtype=tf.float32))
        logp_h_given_z = bernoulli_log_likelihood(h, mu)
        return logp_z + logp_h_given_z, z, h

    # exact ll when dim_z is small
    def logprobs(self, h):
        assert (self.dims[0] <= 15)
        z_all = np.arange(2**self.dims[0], dtype=np.int32)
        z_all = ((z_all.reshape(-1, 1) &
                  (2**np.arange(self.dims[0]))) != 0).astype(np.float32)
        z_all = tf.constant(z_all[:, ::-1], dtype=tf.float32)
        h_mu_all = self.inference(z_all)

        logp_z_all = bernoulli_log_likelihood(
            z_all,
            tf.constant(
                .5, shape=(2**self.dims[0], self.dims[0]), dtype=tf.float32))
        logp_h_given_z_all = bernoulli_log_likelihood(
            tf.expand_dims(h, 1), tf.expand_dims(h_mu_all, 0))
        logp_h = tf.reduce_logsumexp(
            tf.expand_dims(logp_z_all, 0) + logp_h_given_z_all, axis=1)

        return logp_h

    def params(self):
        return tuple(self.ws + self.bs)


class SBNDEC1():

    def __init__(self,
                 dim_zh2,
                 dims_h2h1,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h2 = SBNPrior(
            dim_zh2,
            name='zh2',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h2h1 = DEC(
            dims_h2h1,
            name='h2h1',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)

    def get_h(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        return h2.numpy()

    # for visualization
    def get_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_hard(h2)
        return h1.numpy()

    def get_v_mean(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.inference(h2)
        return h1.numpy()

    # for training
    def get_h1_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_soft(h2)
        return h2, h1

    def log_prob(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        logp_h2 = self.prior_h2.logprobs(h2)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        return logp_h2 + logp_h1_given_h2, h2, h1

    def log_prob_all(self, num_samples):
        logp_z_and_h2, z, h2 = self.prior_h2.logprobs_all(num_samples)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        return logp_z_and_h2 + logp_h1_given_h2, z, h2, h1

    def params(self):
        return self.prior_h2.params() + self.dec_h2h1.params()


class SBNDEC2():

    def __init__(self,
                 dim_zh2,
                 dims_h2h1,
                 dims_h1v,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h2 = SBNPrior(
            dim_zh2,
            name='zh2',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h2h1 = DEC(
            dims_h2h1,
            name='h2h1',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h1v = DEC(
            dims_h1v,
            name='h1v',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)

    # for visualization
    def get_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_hard(h2)
        v = self.dec_h1v.get_sample_hard(h1)
        return v.numpy()

    def get_v_mean(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_hard(h2)
        v = self.dec_h1v.inference(h1)
        return v.numpy()

    # for training
    def get_h2_h1_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_soft(h2)
        v = self.dec_h1v.get_sample_soft(h1)
        return h2, h1, v

    def log_prob(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        logp_h2 = self.prior_h2.logprobs(h2)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        logp_v_given_h1, v = self.dec_h1v.log_conditional_prob(h1)
        return logp_v_given_h1 + logp_h1_given_h2 + logp_h2, h2, h1, v

    def log_prob_all(self, num_samples):
        logp_z_and_h2, z, h2 = self.prior_h2.logprobs_all(num_samples)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        logp_v_given_h1, v = self.dec_h1v.log_conditional_prob(h1)
        return logp_v_given_h1 + logp_z_and_h2 + logp_h1_given_h2, z, h2, h1, v

    def params(self):
        return self.prior_h2.params() + self.dec_h1v.params(
        ) + self.dec_h2h1.params()


class BNDEC():

    def __init__(self,
                 dims,
                 name,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False):
        self.dims = dims
        self.ws, self.scales, self.shifts = [], [], []

        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        assert len(self.dims) > 1
        for d1, d2, i in zip(dims[:-1], dims[1:], xrange(1, len(self.dims))):
            self.ws.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='enc.' + name + '.w.' + str(i)))
            self.scales.append(
                tfe.Variable(
                    tf.ones([d2]),
                    dtype=tf.float32,
                    name='enc.' + name + '.scale.' + str(i)))
            self.shifts.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='enc.' + name + '.shift.' + str(i)))

        self.activation_fn = activation_fn
        self.temp = temp
        self.hard = hard

    def inference(self, x):
        h = x
        for w, sc, sh, i in zip(self.ws, self.scales, self.shifts,
                                xrange(len(self.ws))):
            h = tf.matmul(h, w)
            mean, var = tf.nn.moments(h, axes=[0])

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([mean, var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mean), tf.identity(var)

            mean, var = mean_var_with_update()

            h = tf.nn.batch_normalization(h, mean, var, sh, sc, 0.001)

            if i == len(self.ws) - 1:
                h = tf.nn.sigmoid(h)
            else:
                h = self.activation_fn(h)
        return h

    def log_conditional_prob(self, x):
        # E_{Q(h|x)} log Q(h|x)
        h_mu = self.inference(x)
        h = gumbel_sigmoid_sample(h_mu, self.temp, self.hard)
        logp_h_given_x = bernoulli_log_likelihood(h, h_mu)
        return logp_h_given_x, h

    def get_sample_hard(self, x):
        return sample_from_bernoulli(self.inference(x))

    def get_sample_soft(self, x):
        return gumbel_sigmoid_sample(self.inference(x), self.temp, self.hard)

    def params(self):
        return tuple(self.ws + self.scales + self.shifts)


class DEC():

    def __init__(self,
                 dims,
                 name,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False):
        self.dims = dims
        self.ws, self.bs = [], []

        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        assert len(self.dims) > 1
        for d1, d2, i in zip(dims[:-1], dims[1:], xrange(1, len(self.dims))):
            self.ws.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='dec.' + name + '.w.' + str(i)))
            self.bs.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='dec.' + name + '.b.' + str(i)))

        self.activation_fn = activation_fn
        self.temp = temp
        self.hard = hard

    def inference(self, x):
        h = x
        for w, b, i in zip(self.ws, self.bs, xrange(len(self.bs))):
            h = tf.matmul(h, w) + b
            if i == len(self.ws) - 1:
                h = tf.nn.sigmoid(h)
            else:
                h = self.activation_fn(h)
        return h

    def log_conditional_prob(self, x):
        # E_{Q(h|x)} log Q(h|x)
        h_mu = self.inference(x)
        h = gumbel_sigmoid_sample(h_mu, self.temp, self.hard)
        logp_h_given_x = bernoulli_log_likelihood(h, h_mu)
        return logp_h_given_x, h

    def log_conditional_prob_with_sampels(self, x, h):
        h_mu = self.inference(x)
        logp_h_given_x = bernoulli_log_likelihood(h, h_mu)
        return logp_h_given_x

    def get_sample_hard(self, x):
        return sample_from_bernoulli(self.inference(x))

    def get_sample_soft(self, x):
        return gumbel_sigmoid_sample(self.inference(x), self.temp, self.hard)

    def params(self):
        return tuple(self.ws + self.bs)


class NADEDEC():

    def __init__(self,
                 dim_zh2,
                 dims_h2h1,
                 dims_h1v,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h2 = NADE(dim_zh2[1], dim_zh2[0], temp)
        self.dec_h2h1 = DEC(
            dims_h2h1,
            name='h2h1',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h1v = DEC(
            dims_h1v,
            name='h1v',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)

    # for visualization
    def get_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_hard(h2)
        v = self.dec_h1v.get_sample_hard(h1)
        return v.numpy()

    # for training
    def get_h2_h1_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_soft(h2)
        v = self.dec_h1v.get_sample_soft(h1)
        return h2, h1, v

    def log_prob(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        logp_h2 = self.prior_h2.logprobs(h2)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        logp_v_given_h1, v = self.dec_h1v.log_conditional_prob(h1)
        return logp_v_given_h1 + logp_h1_given_h2 + logp_h2, h2, h1, v

    def params(self):
        return self.prior_h2.params() + self.dec_h1v.params(
        ) + self.dec_h2h1.params()


class SBN():

    def __init__(self,
                 dim_z2h,
                 dim_h2v,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 has_h=True):
        self.dim_z2h = dim_z2h
        self.w_z2h, self.b_z2h = [], []

        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        if has_h:
            assert len(self.dim_z2h) > 1
            for d1, d2, i in zip(dim_z2h[:-1], dim_z2h[1:],
                                 xrange(1, len(self.dim_z2h))):
                self.w_z2h.append(
                    tfe.Variable(
                        tf_xavier_init(d1, d2, const=const),
                        name='sbn.w_z2h.' + str(i)))
                self.b_z2h.append(
                    tfe.Variable(
                        tf.zeros([d2]),
                        dtype=tf.float32,
                        name='sbn.b_z2h.' + str(i)))

        self.dim_h2v = dim_h2v
        self.w_h2v, self.b_h2v = [], []
        assert len(self.dim_h2v) > 1
        for d1, d2, i in zip(dim_h2v[:-1], dim_h2v[1:],
                             xrange(1, len(self.dim_h2v))):
            self.w_h2v.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='sbn.w_h2v.' + str(i)))
            self.b_h2v.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='sbn.b_h2v.' + str(i)))

        self.activation_fn = activation_fn
        self.temp = temp
        self.hard = hard
        self.has_h = has_h  # Turning it false to train a 2 layer sbn for debugging, otherwise always true

    def z2h(self, z):
        h = z
        for w, b, i in zip(self.w_z2h, self.b_z2h, xrange(len(self.w_z2h))):
            h = tf.matmul(h, w) + b
            if i == len(self.w_z2h) - 1:
                h = tf.nn.sigmoid(h)
            else:
                h = self.activation_fn(h)
        return h

    def h2v(self, h):
        v = h
        for w, b, i in zip(self.w_h2v, self.b_h2v, xrange(len(self.w_h2v))):
            v = tf.matmul(v, w) + b
            if i == len(self.w_h2v) - 1:
                v = tf.nn.sigmoid(v)
            else:
                v = self.activation_fn(v)
        return v

    # for visualization
    def get_v(self, num_samples):
        z = sample_from_bernoulli(
            tf.constant(
                .5, shape=(num_samples, self.dim_z2h[0]), dtype=tf.float32))
        h = sample_from_bernoulli(self.z2h(z))
        v = sample_from_bernoulli(self.h2v(h))
        return v.numpy()

    # only for debugging, get_samples from 2 layer sbn
    def _get_hard_v(self, num_samples):
        h = sample_from_bernoulli(
            tf.constant(
                .5, shape=(num_samples, self.dim_h2v[0]), dtype=tf.float32))
        v = sample_from_bernoulli(self.h2v(h))
        return v.numpy()

    # for training
    def get_h_and_v(self, num_samples):
        z = sample_from_bernoulli(
            tf.constant(
                .5, shape=(num_samples, self.dim_z2h[0]), dtype=tf.float32))
        h = gumbel_sigmoid_sample(self.z2h(z), self.temp, self.hard)
        v = gumbel_sigmoid_sample(self.h2v(h), self.temp, self.hard)
        return h, v

    # only for debugging, testing gumbel_softmax
    def _test_gumbel(self, num_samples=1):
        z = sample_from_bernoulli(
            tf.constant(
                .5, shape=(num_samples, self.dim_z2h[0]), dtype=tf.float32))
        h_mu = self.z2h(z)
        h_hard = gumbel_sigmoid_sample(h_mu, self.temp, True)
        h_soft = gumbel_sigmoid_sample(h_mu, self.temp, False)
        print 'h_mu\n', h_mu, 'h_hard\n', h_hard, 'h_soft\n', h_soft
        v_mu = self.h2v(h_hard)
        v_hard = gumbel_sigmoid_sample(v_mu, self.temp, True)
        v_soft = gumbel_sigmoid_sample(v_mu, self.temp, False)
        print 'v_mu\n', v_mu, 'v_hard\n', v_hard, 'v_soft\n', v_soft
        v_mu = self.h2v(h_soft)
        v_hard = gumbel_sigmoid_sample(v_mu, self.temp, True)
        v_soft = gumbel_sigmoid_sample(v_mu, self.temp, False)
        print 'v_mu\n', v_mu, 'v_hard\n', v_hard, 'v_soft\n', v_soft
        exit()

    # exact ll when dim_z is small
    def logprob_v_and_h(self, h, v):
        assert (self.dim_z2h[0] <= 15)
        z_all = np.arange(2**self.dim_z2h[0], dtype=np.int32)
        z_all = ((z_all.reshape(-1, 1) &
                  (2**np.arange(self.dim_z2h[0]))) != 0).astype(np.float32)
        z_all = tf.constant(z_all[:, ::-1], dtype=tf.float32)
        h_mu_all = self.z2h(z_all)

        logp_z_all = bernoulli_log_likelihood(
            z_all,
            tf.constant(
                .5,
                shape=(2**self.dim_z2h[0], self.dim_z2h[0]),
                dtype=tf.float32))
        logp_h_given_z_all = bernoulli_log_likelihood(
            tf.expand_dims(h, 1), tf.expand_dims(h_mu_all, 0))
        logp_h = tf.reduce_logsumexp(
            tf.expand_dims(logp_z_all, 0) + logp_h_given_z_all, axis=1)
        logp_v_given_h = bernoulli_log_likelihood(v, self.h2v(h))

        return logp_v_given_h + logp_h

    def logprob_v_and_h_approx(self, h, v):
        h_mu = tf.reduce_mean(h, axis=0)
        logp_h = bernoulli_log_likelihood(h, h_mu)
        logp_v_given_h = bernoulli_log_likelihood(v, self.h2v(h))
        return logp_v_given_h + logp_h

    def logprob_v_and_h_and_z(self, v, h, z):
        logp_z = bernoulli_log_likelihood(
            z, tf.constant(.5, shape=z.shape, dtype=tf.float32))
        logp_h_given_z = bernoulli_log_likelihood(h, self.z2h(z))
        logp_v_given_h = bernoulli_log_likelihood(v, self.h2v(h))
        return logp_v_given_h + logp_h_given_z + logp_z

    # only for debugging, exact ll of 2 layer sbn
    def _logprob_v(self, h):
        assert (self.dim_h2v[0] <= 15)
        assert (self.has_h == False)
        z_all = np.arange(2**self.dim_h2v[0], dtype=np.int32)
        z_all = ((z_all.reshape(-1, 1) &
                  (2**np.arange(self.dim_h2v[0]))) != 0).astype(np.float32)
        z_all = tf.constant(z_all[:, ::-1], dtype=tf.float32)
        h_mu_all = self.h2v(z_all)
        #print z_all.numpy().shape, h_mu_all.numpy().shape

        logp_z_all = bernoulli_log_likelihood(
            z_all,
            tf.constant(
                .5,
                shape=(2**self.dim_h2v[0], self.dim_h2v[0]),
                dtype=tf.float32))
        #print logp_z_all.numpy().shape
        logp_h_given_z_all = bernoulli_log_likelihood(
            tf.expand_dims(h, 1), tf.expand_dims(h_mu_all, 0))
        #print logp_h_given_z_all.numpy().shape
        logp_h = tf.reduce_logsumexp(
            tf.expand_dims(logp_z_all, 0) + logp_h_given_z_all, axis=1)
        return logp_h

    def params(self):
        return tuple(self.w_z2h + self.b_z2h + self.w_h2v + self.b_h2v)


class MOB():

    def __init__(self, dim, num, temp=.1, hard=False, train_pi=True):
        self.num = num
        self.dim = dim
        self.train_pi = train_pi
        if self.train_pi:
            self.pi_logits = tfe.Variable(
                tf.zeros([1, self.num]),
                dtype=tf.float32,
                name='dec.mob.pi_logits')
        else:
            self.pi_logits = tf.constant(
                .0, shape=[1, self.num], dtype=tf.float32)
        self.mus = tfe.Variable(
            tf.ones([self.num, self.dim]) * .5,
            dtype=tf.float32,
            name='dec.mob.mus')
        self.temp = temp
        self.hard = hard

    def logprobs(self, x):
        # log \sum_k pi_k p(x|mu_k)
        # log sum exp_k (log pi_k + log p(x|mu_k))
        log_pi = self.pi_logits - tf.reduce_logsumexp(self.pi_logits, axis=1)
        logp_x_given_mu_all = bernoulli_log_likelihood(
            tf.expand_dims(x, 1), tf.expand_dims(self.mus, 0))
        return tf.reduce_logsumexp(log_pi + logp_x_given_mu_all, axis=1)

    def sample(self, num_samples):
        index = gumbel_softmax_sample(
            tf.tile(self.pi_logits, [num_samples, 1]), self.temp)
        sample_mus = tf.matmul(index, self.mus)
        return gumbel_sigmoid_sample(sample_mus, self.temp, self.hard)

    def params(self):
        if self.train_pi:
            return (self.pi_logits, self.mus)
        else:
            return (self.mus,)


class MOBSBN():

    def __init__(self,
                 num_mixtures,
                 dim_h2v,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h = MOB(dim_h2v[0], num_mixtures, temp, hard, train_pi)
        self.dim_h2v = dim_h2v
        self.w_h2v, self.b_h2v = [], []
        assert len(self.dim_h2v) > 1
        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        for d1, d2, i in zip(dim_h2v[:-1], dim_h2v[1:],
                             xrange(1, len(self.dim_h2v))):
            self.w_h2v.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='sbn.w_h2v.' + str(i)))
            self.b_h2v.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='sbn.b_h2v.' + str(i)))

        self.activation_fn = activation_fn
        self.temp = temp
        self.hard = hard

    def h2v(self, h):
        v = h
        for w, b, i in zip(self.w_h2v, self.b_h2v, xrange(len(self.w_h2v))):
            v = tf.matmul(v, w) + b
            if i == len(self.w_h2v) - 1:
                v = tf.nn.sigmoid(v)
            else:
                v = self.activation_fn(v)
        return v

    # for visualization
    def get_v(self, num_samples):
        h = self.prior_h.sample(num_samples)
        v = sample_from_bernoulli(self.h2v(h))
        return v.numpy()

    # for training
    def get_h_and_v(self, num_samples):
        h = self.prior_h.sample(num_samples)
        v = gumbel_sigmoid_sample(self.h2v(h), self.temp, self.hard)
        return h, v

    def logprob_v_and_h(self, h, v):
        logp_h = self.prior_h.logprobs(h)
        logp_v_given_h = bernoulli_log_likelihood(v, self.h2v(h))
        return logp_v_given_h + logp_h

    def params(self):
        return self.prior_h.params() + tuple(self.w_h2v + self.b_h2v)


class NADESBN():

    def __init__(self,
                 dim_z2h,
                 dim_h2v,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h = NADE(dim_z2h[1], dim_z2h[0], temp)
        self.dim_h2v = dim_h2v
        self.w_h2v, self.b_h2v = [], []
        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        assert len(self.dim_h2v) > 1
        for d1, d2, i in zip(dim_h2v[:-1], dim_h2v[1:],
                             xrange(1, len(self.dim_h2v))):
            self.w_h2v.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='sbn.w_h2v.' + str(i)))
            self.b_h2v.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='sbn.b_h2v.' + str(i)))

        self.activation_fn = activation_fn
        self.temp = temp
        self.hard = hard

    def h2v(self, h):
        v = h
        for w, b, i in zip(self.w_h2v, self.b_h2v, xrange(len(self.w_h2v))):
            v = tf.matmul(v, w) + b
            if i == len(self.w_h2v) - 1:
                v = tf.nn.sigmoid(v)
            else:
                v = self.activation_fn(v)
        return v

    def get_h(self, num_samples):
        h = self.prior_h.sample(num_samples)
        return h.numpy()

    # for visualization
    def get_v(self, num_samples):
        h = self.prior_h.sample(num_samples)
        v = sample_from_bernoulli(self.h2v(h))
        return v.numpy()

    # for training
    def get_h_and_v(self, num_samples):
        h = self.prior_h.sample(num_samples)
        v = gumbel_sigmoid_sample(self.h2v(h), self.temp, self.hard)
        return h, v

    def logprob_v_and_h(self, h, v):
        logp_h = self.prior_h.logprobs(h)
        logp_v_given_h = bernoulli_log_likelihood(v, self.h2v(h))
        return logp_v_given_h + logp_h

    def params(self):
        return self.prior_h.params() + tuple(self.w_h2v + self.b_h2v)


class VAEDEC():
    def __init__(self, dim_z2x, activation_fn=tf.nn.tanh, temp=.1, hard=False):
        self.dec_z2x = DEC(
            dim_z2x,
            name='z2x',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.z_dim = dim_z2x[0]

    # for visualization
    def get_v(self, num_samples):
        h2 = tf.random_normal((num_samples, self.z_dim))
        h1 = self.dec_z2x.get_sample_hard(h2)
        return h1.numpy()

    def get_v_mean(self, num_samples):
        h2 = tf.random_normal((num_samples, self.z_dim))
        h1 = self.dec_z2x.inference(h2)
        return h1.numpy()

    # for training
    def get_h1_v(self, num_samples):
        h2 = tf.random_normal((num_samples, self.z_dim))
        h1 = self.dec_z2x.get_sample_soft(h2)
        return h2, h1

    def log_prob(self, num_samples):
        h2 = tf.random_normal((num_samples, self.z_dim))
        logp_h2 = tf.reduce_sum(
            -0.5 * np.log(2 * np.pi) - 0.5 * h2 * h2, axis=-1)
        logp_h1_given_h2, h1 = self.dec_z2x.log_conditional_prob(h2)
        return logp_h2 + logp_h1_given_h2, h2, h1

    def get_v_tf(self, num_samples):
        h2 = tf.random_normal((num_samples, self.z_dim))
        h1 = self.dec_z2x.get_sample_hard(h2)
        return h1

    def log_cond_prob(self, z, x):
        logp_h1_given_h2 = self.dec_z2x.log_conditional_prob_given(z, x)
        return logp_h1_given_h2

    def params(self):
        return self.dec_z2x.params()


class NADE_q():

    def __init__(self, num_dims, num_hidden, temp=.1, hard=False):
        self.nade = NADE(num_dims, num_hidden, temperature=temp, hard=hard)

    # for visualization
    def get_v(self, num_samples):
        return self.nade.sample(num_samples).numpy()

    def log_prob(self, num_samples):
        v = self.nade.sample(num_samples)
        logp_v = self.nade.logprobs(v)
        return logp_v, v

    def params(self):
        return self.nade.params()


class VAEDEC_ZHV():

    def __init__(self,
                 dim_zh2,
                 dims_h2h1,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h2 = VAEDEC(
            dim_zh2,
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h2h1 = DEC(
            dims_h2h1,
            name='h2h1',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)

    def get_h(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        return h2.numpy()

    # for visualization
    def get_v(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.get_sample_hard(h2)
        return h1.numpy()

    def get_v_mean(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.inference(h2)
        return h1.numpy()

    # for training
    def get_h1_v(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.get_sample_soft(h2)
        return h2, h1

    def log_prob_all(self, num_samples):
        logp_z_and_h2, z, h2 = self.prior_h2.log_prob(num_samples)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        return logp_z_and_h2 + logp_h1_given_h2, z, h2, h1

    def params(self):
        return self.prior_h2.params() + self.dec_h2h1.params()


class VAEDEC_ZHVC():

    def __init__(self,
                 dim_zh2,
                 dims_h2h1,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h2 = VAEDEC(
            dim_zh2,
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h2h1 = ENC_CONT(
            dims_h2h1,
            name='h2h1',
            activation_fn=activation_fn)

    def get_h(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        return h2.numpy()

    # for visualization
    def get_v(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.get_h(h2)
        return h1.numpy()

    def get_v_mean(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1, _ = self.dec_h2h1.inference(h2)
        return h1.numpy()

    # for training
    def get_h1_v(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.get_h(h2)
        return h2, h1

    def log_prob_all(self, num_samples):
        logp_z_and_h2, z, h2 = self.prior_h2.log_prob(num_samples)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        return logp_z_and_h2 + logp_h1_given_h2, z, h2, h1

    def params(self):
        return self.prior_h2.params() + self.dec_h2h1.params()


class VAEDEC_ZHV2():

    def __init__(self,
                 dim_zh2,
                 dims_h2h1,
                 dims_h1v,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False,
                 train_pi=True):
        self.prior_h2 = VAEDEC(
            dim_zh2,
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h2h1 = DEC(
            dims_h2h1,
            name='h2h1',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)
        self.dec_h1v = DEC(
            dims_h1v,
            name='h1v',
            activation_fn=activation_fn,
            temp=temp,
            hard=hard)

    def get_h(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        return h2.numpy()

    # for visualization
    def get_v(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.get_sample_hard(h2)
        v = self.dec_h1v.get_sample_hard(h1)
        return v.numpy()

    def get_v_mean(self, num_samples):
        h2 = self.prior_h2.get_v_tf(num_samples)
        h1 = self.dec_h2h1.inference(h2)
        v = self.dec_h1v.inference(h1)
        return v.numpy()

    # for training
    def get_h2_h1_v(self, num_samples):
        h2 = self.prior_h2.sample(num_samples)
        h1 = self.dec_h2h1.get_sample_soft(h2)
        v = self.dec_h1v.get_sample_soft(h1)
        return h2, h1, v

    def log_prob_all(self, num_samples):
        logp_z_and_h2, z, h2 = self.prior_h2.log_prob(num_samples)
        logp_h1_given_h2, h1 = self.dec_h2h1.log_conditional_prob(h2)
        logp_v_given_h1, v = self.dec_h1v.log_conditional_prob(h1)
        return logp_v_given_h1 + logp_z_and_h2 + logp_h1_given_h2, z, h2, h1, v

    def params(self):
        return self.prior_h2.params() + self.dec_h2h1.params() + self.dec_h1v.params()
