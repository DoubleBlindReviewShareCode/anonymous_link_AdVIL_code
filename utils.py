import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave
import imageio

# Xavier Glorot init


def tf_xavier_init(fan_in, fan_out, const=1.0):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=np.float32)

# Sample from Gumbel(0, 1)


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

# Draw a sample from the Gumbel-Softmax distribution


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

# Draw a sample from the Gumbel-Sigmoid distribution


def gumbel_sigmoid_sample_logits(logits, temperature, hard):
    shape = logits.shape
    logits = tf.reshape(logits, [-1, 1])
    logits = tf.concat([logits, tf.zeros(logits.shape)], axis=1)
    softmax_sample = gumbel_softmax_sample(logits, temperature)
    sigmoid_sample = softmax_sample[:, 0]
    if hard:
        softmax_sample_hard = tf.cast(tf.equal(softmax_sample, tf.reduce_max(softmax_sample, 1, keep_dims=True)), softmax_sample.dtype)
        sigmoid_sample_hard = softmax_sample_hard[:, 0]
        # check gumbel softmax
        # print sigmoid_sample
        # print sigmoid_sample_hard
        # print softmax_sample[:,1]
        # print softmax_sample_hard[:,1]
        # exit()
        sigmoid_sample = tf.stop_gradient(sigmoid_sample_hard - sigmoid_sample) + sigmoid_sample
    return tf.reshape(sigmoid_sample, shape)

# Draw a sample from the Gumbel-Sigmoid distribution


def gumbel_sigmoid_sample(probs, temperature, hard, eps=1e-5):
    shape = probs.shape
    probs = tf.clip_by_value(tf.reshape(probs, [-1, 1]), eps, 1 - eps)
    logits = tf.concat([tf.log(probs / (1 - probs)), tf.zeros(probs.shape)], axis=1)
    softmax_sample = gumbel_softmax_sample(logits, temperature)
    sigmoid_sample = softmax_sample[:, 0]
    if hard:
        softmax_sample_hard = tf.cast(tf.equal(softmax_sample, tf.reduce_max(softmax_sample, 1, keep_dims=True)), softmax_sample.dtype)
        sigmoid_sample_hard = softmax_sample_hard[:, 0]
        # check gumbel softmax
        # print sigmoid_sample
        # print sigmoid_sample_hard
        # print softmax_sample[:,1]
        # print softmax_sample_hard[:,1]
        # exit()
        sigmoid_sample = tf.stop_gradient(sigmoid_sample_hard - sigmoid_sample) + sigmoid_sample
    return tf.reshape(sigmoid_sample, shape)

# Sample from Bernoulli distribution


def sample_from_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def sample_from_gaussian(mu, logvar):
    return tf.random_normal(tf.shape(mu)) * tf.exp(0.5 * logvar) + mu

# Bernoulli likelihood


def bernoulli_log_likelihood(labels, probs, eps=1e-6):
    probs = tf.clip_by_value(probs, eps, 1 - eps)
    return tf.reduce_sum(labels * tf.log(probs) + (1 - labels) * tf.log(1 - probs), axis=-1)

# Image grid saver, based on color_grid_vis from github.com/Newmu


def large_image(X, size=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]

    if size is None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples / rows
    else:
        nh, nw = size
        assert(nh * nw == n_samples)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    return img.astype('uint8')


def save_gifs(x, save_path, size=None):
    final_list = []
    for i in xrange(x.shape[1]):
        final_list.append(large_image(x[:, i, :, :, :], size=size))
    imageio.mimsave(save_path, final_list)


def save_images(X, save_path, size=None, normalize=False, mg=1):
    # arbitary to [0, 1]
    if normalize:
        _min = np.min(X)
        X = X - _min
        _max = np.max(X)
        if _max == 0:
            _max = 1
        X = X / _max
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]

    if size is None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples / rows
    else:
        nh, nw = size
        assert(nh * nw == n_samples)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros(((h + mg) * nh - mg, (w + mg) * nw - mg, 3)) + 127.
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros(((h + mg) * nh - mg, (w + mg) * nw - mg)) + 127.

    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * (h + mg):j * (h + mg) + h, i * (w + mg):i * (w + mg) + w] = x

    imsave(save_path, img)


plt.switch_backend('Agg')


def visualize_latent_space(samples, dim, outf, name):
    temp = [2**(dim - 1)]
    while temp[-1] > 1:
        temp.append(temp[-1] / 2)
    temp = np.tile(np.asarray(temp), (samples.shape[0], 1))
    indices = np.sum(temp * samples, axis=1)
    n, bins, patches = plt.hist(indices, 2**dim, facecolor='red', alpha=0.75)
    # legend = plt.legend(('neg_loss', 'ais LL', 'dec loss', 'dec entropy', 'pos free energy', 'rbm loss', 'ais log Z'))
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(os.path.join(outf, name + '.png'), bbox_inches='tight')
    plt.close()
