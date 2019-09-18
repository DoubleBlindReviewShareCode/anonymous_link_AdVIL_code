"""
Neural autoregressive density estimator (NADE) implementation
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import gumbel_sigmoid_sample_logits

def _safe_log(tensor):
  return tf.log(1e-6 + tensor)

class NADE():
    def __init__(self, num_dims, num_hidden, temperature=.1, name='dec.nade.', hard=False):
        self.num_dims = num_dims
        self.num_hidden = num_hidden
        self.temperature = temperature
        self.hard = hard
        std = 1.0 / math.sqrt(self.num_dims)
        initializer = tf.truncated_normal_initializer(stddev=std)
        self.w_enc = tfe.Variable(initializer(shape=[self.num_dims, 1, self.num_hidden]), name=name+'w_enc')
        self.w_dec_t = tfe.Variable(initializer(shape=[self.num_dims, self.num_hidden, 1]), name=name+'w_dec_t')
        self.b_enc = tfe.Variable(initializer(shape=[1, self.num_hidden]), name=name+'b_enc')
        self.b_dec = tfe.Variable(initializer(shape=[1, self.num_dims]), name=name+'b_dec')

    def params(self):
        return (self.w_enc, self.w_dec_t, self.b_enc, self.b_dec)

    def logprobs(self, x):
        batch_size = tf.shape(x)[0]
        b_enc = self.b_enc
        b_dec = self.b_dec
        if b_enc.shape[0] == 1 != batch_size:
            b_enc = tf.tile(b_enc, [batch_size, 1])
        if b_dec.shape[0] == 1 != batch_size:
            b_dec = tf.tile(b_dec, [batch_size, 1])
        
        a_0 = b_enc
        log_p_0 = tf.zeros([batch_size, 1])
        cond_p_0 = []

        x_arr = tf.unstack(tf.reshape(tf.transpose(x), [self.num_dims, batch_size, 1]))
        w_enc_arr = tf.unstack(self.w_enc)
        w_dec_arr = tf.unstack(self.w_dec_t)
        b_dec_arr = tf.unstack(tf.reshape(tf.transpose(b_dec), [self.num_dims, batch_size, 1]))

        def loop_body(i, a, log_p, cond_p):
            # Get variables for time step.
            w_enc_i = w_enc_arr[i]
            w_dec_i = w_dec_arr[i]
            b_dec_i = b_dec_arr[i]
            v_i = x_arr[i]

            cond_p_i, _ = self._cond_prob(a, w_dec_i, b_dec_i)

            # Get log probability for this value. Log space avoids numerical issues.
            log_p_i = v_i * _safe_log(cond_p_i) + (1 - v_i) * _safe_log(1 - cond_p_i)

            # Accumulate log probability.
            log_p_new = log_p + log_p_i

            # Save conditional probabilities.
            cond_p_new = cond_p + [cond_p_i]

            # Encode value and add to hidden units.
            a_new = a + tf.matmul(v_i, w_enc_i)

            return a_new, log_p_new, cond_p_new

        # Build the actual loop
        a, log_p, cond_p = a_0, log_p_0, cond_p_0
        for i in range(self.num_dims):
            a, log_p, cond_p = loop_body(i, a, log_p, cond_p)
        
        return tf.squeeze(log_p, squeeze_dims=[1])

    def sample(self, n=None):    
        b_enc = self.b_enc
        b_dec = self.b_dec
        batch_size = n or tf.shape(b_enc)[0]
        # Broadcast if needed.
        if b_enc.shape[0] == 1 != batch_size:
            b_enc = tf.tile(b_enc, [batch_size, 1])
        if b_dec.shape[0] == 1 != batch_size:
            b_dec = tf.tile(b_dec, [batch_size, 1])
        a_0 = b_enc
        sample_0 = []
        log_p_0 = tf.zeros([batch_size, 1])
        logit_0 = tf.zeros([batch_size, 1])

        w_enc_arr = tf.unstack(self.w_enc)
        w_dec_arr = tf.unstack(self.w_dec_t)
        b_dec_arr = tf.unstack(tf.reshape(tf.transpose(b_dec), [self.num_dims, batch_size, 1]))

        def loop_body(i, a, sample, log_p):
            w_enc_i = w_enc_arr[i]
            w_dec_i = w_dec_arr[i]
            b_dec_i = b_dec_arr[i]

            cond_p_i, cond_l_i = self._cond_prob(a, w_dec_i, b_dec_i)

            if self.hard:
                bernoulli = tf.distributions.Bernoulli(logits=cond_l_i / self.temperature, dtype=tf.float32)
                v_i = bernoulli.sample()
            else:
                v_i = gumbel_sigmoid_sample_logits(cond_l_i, self.temperature, False)

            # Accumulate sampled values.
            sample_new = sample + [v_i]
            logit_new = logit + [cond_l_i]

            # Get log probability for this value. Log space avoids numerical issues.
            log_p_i = v_i * _safe_log(cond_p_i) + (1 - v_i) * _safe_log(1 - cond_p_i)

            # Accumulate log probability.
            log_p_new = log_p + log_p_i

            # Encode value and add to hidden units.
            a_new = a + tf.matmul(v_i, w_enc_i)

            return a_new, sample_new, log_p_new, logit_new

        a, sample, log_p, logit = a_0, sample_0, log_p_0, logit_0
        for i in range(self.num_dims):
            a, sample, log_p, logit = loop_body(i, a, sample, log_p)
        if self.hard:
            logit = tf.transpose(tf.squeeze(tf.stack(logit), [2]))
            sample = gumbel_sigmoid_sample_logits(logit, self.temperature, False)
        else:
            sample = tf.transpose(tf.squeeze(tf.stack(sample), [2]))
        return sample

    # Decode hidden units to get conditional probability.
    def _cond_prob(self, a, w_dec_i, b_dec_i):
        h = tf.sigmoid(a)
        cond_l_i = b_dec_i + tf.matmul(h, w_dec_i)
        cond_p_i = tf.sigmoid(cond_l_i)
        return cond_p_i, cond_l_i