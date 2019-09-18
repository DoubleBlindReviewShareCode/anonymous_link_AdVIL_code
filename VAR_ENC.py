'''
Probabilistic encoders as variational approximation for inference
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import gumbel_sigmoid_sample, sample_from_bernoulli, bernoulli_log_likelihood, tf_xavier_init, sample_from_gaussian


class SBNENC():

    def __init__(self,
                 dims_vh1,
                 dims_h1h2,
                 BN=False,
                 activation_fn=tf.nn.relu,
                 temp=.1,
                 hard=False):
        if BN is False:
            self.enc_vh1 = ENC(
                dims_vh1,
                name='vh1',
                activation_fn=activation_fn,
                temp=temp,
                hard=hard)
            self.enc_h1h2 = ENC(
                dims_h1h2,
                name='h1h2',
                activation_fn=activation_fn,
                temp=temp,
                hard=hard)
        else:
            self.enc_vh1 = BNENC(
                dims_vh1,
                name='vh1',
                activation_fn=activation_fn,
                temp=temp,
                hard=hard)
            self.enc_h1h2 = BNENC(
                dims_h1h2,
                name='h1h2',
                activation_fn=activation_fn,
                temp=temp,
                hard=hard)

    def log_conditional_prob(self, x):
        logp_h1_given_x, h1 = self.enc_vh1.log_conditional_prob(x)
        logp_h2_given_h1, h2 = self.enc_h1h2.log_conditional_prob(h1)
        return logp_h1_given_x + logp_h2_given_h1, h1, h2

    def get_hard_h1(self, x):
        return self.enc_vh1.get_h_hard(x)

    def params(self):
        return self.enc_vh1.params() + self.enc_h1h2.params()


class ENC():

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
                    name='enc.' + name + '.w.' + str(i)))
            self.bs.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='enc.' + name + '.b.' + str(i)))

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

    def log_conditional_prob_evaluate(self, x, h):
        # E_{Q(h|x)} log Q(h|x)
        h_mu = self.inference(x)
        logp_h_given_x = bernoulli_log_likelihood(h, h_mu)
        return logp_h_given_x

    def get_h_hard(self, x):
        return sample_from_bernoulli(self.inference(x))

    def get_h_soft(self, x):
        return gumbel_sigmoid_sample(self.inference(x), self.temp, self.hard)

    def params(self):
        return tuple(self.ws + self.bs)


class BNENC():

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
            mean, var = ema.apply([mean, var])

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

    def log_conditional_prob_evaluate(self, x, h):
        # E_{Q(h|x)} log Q(h|x)
        h_mu = self.inference(x)
        logp_h_given_x = bernoulli_log_likelihood(h, h_mu)
        return logp_h_given_x

    def get_h_hard(self, x):
        return sample_from_bernoulli(self.inference(x))

    def get_h_soft(self, x):
        return gumbel_sigmoid_sample(self.inference(x), self.temp, self.hard)

    def params(self):
        return tuple(self.ws + self.scales + self.shifts)


class ENC_CONT():

    def __init__(self, dims, name, activation_fn=tf.nn.tanh):
        self.dims = dims
        self.ws, self.bs = [], []

        self.activation_fn = activation_fn
        if activation_fn == tf.nn.relu:
            const = 1.0
        else:
            const = 4.0

        assert len(self.dims) > 1
        for d1, d2, i in zip(dims[:-2], dims[1:-1], xrange(
                1,
                len(self.dims) - 1)):
            self.ws.append(
                tfe.Variable(
                    tf_xavier_init(d1, d2, const=const),
                    name='enc.' + name + '.w.' + str(i)))
            self.bs.append(
                tfe.Variable(
                    tf.zeros([d2]),
                    dtype=tf.float32,
                    name='enc.' + name + '.b.' + str(i)))

        self.ws.append(
            tfe.Variable(
                tf_xavier_init(dims[-2], dims[-1], const=const),
                name='enc.' + name + '.w.out.mu'))
        self.bs.append(
            tfe.Variable(
                tf.zeros([dims[-1]]),
                dtype=tf.float32,
                name='enc.' + name + '.b.out.mu'))
        self.ws.append(
            tfe.Variable(
                tf_xavier_init(dims[-2], dims[-1], const=const),
                name='enc.' + name + '.w.out.logvar'))
        self.bs.append(
            tfe.Variable(
                tf.zeros([dims[-1]]),
                dtype=tf.float32,
                name='enc.' + name + '.b.out.logvar'))

    def inference(self, x):
        h = x
        for w, b, i in zip(self.ws[:-2], self.bs[:-2],
                           xrange(len(self.bs) - 2)):
            h = tf.matmul(h, w) + b
            h = self.activation_fn(h)
        h_mu = tf.matmul(h, self.ws[-2]) + self.bs[-2]
        h_logvar = tf.matmul(h, self.ws[-1]) + self.bs[-1]
        return h_mu, h_logvar

    def log_conditional_prob(self, x):
        # E_{Q(h|x)} log Q(h|x)
        h_mu, h_logvar = self.inference(x)
        h = sample_from_gaussian(h_mu, h_logvar)
        logp_h_given_x = tf.reduce_sum(
            -0.5 * np.log(2 * np.pi) - 0.5 * h_logvar -
            0.5 * tf.square(h - h_mu) / tf.exp(h_logvar),
            axis=-1)
        return logp_h_given_x, h

    def log_conditional_prob_evaluate(self, x, h):
        # E_{Q(h|x)} log Q(h|x)
        h_mu, h_logvar = self.inference(x)
        logp_h_given_x = tf.reduce_sum(
            -0.5 * np.log(2 * np.pi) - 0.5 * h_logvar -
            0.5 * tf.square(h - h_mu) / tf.exp(h_logvar),
            axis=-1)
        return logp_h_given_x

    def get_h(self, x):
        h_mu, h_logvar = self.inference(x)
        h = sample_from_gaussian(h_mu, h_logvar)
        return h

    def get_h_hard(self, x):
        return self.get_h(x)

    def get_h_soft(self, x):
        return self.get_h(x)

    def params(self):
        return tuple(self.ws + self.bs)
