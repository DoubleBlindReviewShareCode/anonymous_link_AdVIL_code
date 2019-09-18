import os
import sys
import shutil
import time
import cPickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import dataset
import utils
from RBM import RBM
from AIS import likelihood_ais, estimate_log_partition_function
from VAR_DEC import SBNDEC1
from VAR_ENC import ENC
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

sys.setrecursionlimit(100000)

# set eager API
tfe.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


'''
parameters and data
'''
# algorithm
method = 'avil'  # 'avil'
assert method in ['avil']
# data and model
settings = {
    '8-50-1': (8, 50, 15, [25], [], [], 'sigmoid'),
    '8-50-2': (8, 15, 10, [15], [], [], 'sigmoid'),
}
setting = settings['8-50-2']

pixels = (setting[0],) * 2
vis_dim = pixels[0] * pixels[1]
hid_dim = setting[1]
z_dim = setting[2]
dim_z2h = [z_dim] + setting[3] + [hid_dim]
dim_h2v = [hid_dim] + setting[4] + [vis_dim]
dim_v2h = [vis_dim] + setting[5] + [hid_dim]
hard = False
shuffle_buffer = 10000
# optimization
num_steps = 100000
non_type = setting[6]
if non_type is 'tanh':
    non = tf.nn.tanh
elif non_type is 'sigmoid':
    non = tf.nn.sigmoid
elif non_type is 'relu':
    non = tf.nn.rectify
else:
    print 'unknown nonlinearity'
    exit(0)
opt_type = 'adam'
if opt_type is 'adam':
    lr_rbm = 0.0003
if opt_type is 'mom':
    lr_rbm = 0.05
    lr_end = lr_rbm / 1000
    decay_steps = num_steps / 2
lr_dec = 0.0003
lr_enc = 0.0003
dec_per_rbm = 100
enc_per_rbm = 1
beta_1_rbm = .5
beta_1_dec = .5
beta_1_enc = .5
batch_size = 500
ver_steps = 1000
# evaluation
evaluation_batch_size = 500
num_samples = 100
adjacent_samples = 10
vis_steps = 2000
# log
filename_script = os.path.basename(os.path.realpath(__file__))
outf = os.path.join("result", os.path.splitext(filename_script)[0])
outf += '.'
outf += str(pixels)
outf += '.'
outf += opt_type
outf += '.bs.'
outf += str(batch_size)
outf += '.rbm.'
outf += str([hid_dim, vis_dim])
outf += '.enc.'
outf += str(enc_per_rbm)
outf += '.'
outf += str(dim_v2h)
outf += '.dec.'
outf += str(dec_per_rbm)
outf += '.'
outf += str(dim_z2h)
outf += '.'
outf += str(dim_h2v)
outf += '.'
outf += non_type
outf += '.'
outf += time.strftime("%Y-%m-%d--%H-%M")
if not os.path.exists(outf):
    os.makedirs(outf)
else:
    print 'There exists a same model.'
    exit()
logfile = os.path.join(outf, 'logfile.txt')
shutil.copy(os.path.realpath(__file__), os.path.join(outf, filename_script))

# data
assert (pixels[0] == 8)
train_dataset, validation_dataset, test_dataset = dataset.uci_digits_eval()
train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
dataset_iter = tfe.Iterator(train_dataset)


'''
model and loss
'''
rbm = RBM(vis_dim=vis_dim, hid_dim=hid_dim)
dec = SBNDEC1(dim_z2h, dim_h2v, temp=.1, hard=hard)
enc = ENC(dim_v2h, name='v2h', activation_fn=non, temp=.1, hard=False)
enc_h2z = ENC(dim_z2h[::-1], name='z2h', activation_fn=non, temp=.1, hard=False)


def loss_rbm(real_v):
    _, real_h = enc.log_conditional_prob(real_v)
    positve_phase = tf.reduce_mean(rbm.energy(real_h, real_v), axis=0)
    _, _, fake_h, fake_v = dec.log_prob_all(batch_size)
    negative_phase = tf.reduce_mean(rbm.energy(fake_h, fake_v), axis=0)
    return positve_phase - negative_phase


def loss_dec():
    logprob_v_and_h_and_z, fake_z, fake_h, fake_v = dec.log_prob_all(batch_size)
    negative_phase = tf.reduce_mean(rbm.energy(fake_h, fake_v), axis=0)
    # entropy_term = - tf.reduce_mean(logprob_v_and_h, axis=0)
    logprob_z_given_h = enc_h2z.log_conditional_prob_evaluate(fake_h, fake_z)
    entropy_lb = tf.reduce_mean(logprob_z_given_h - logprob_v_and_h_and_z, axis=0)
    return negative_phase - entropy_lb, negative_phase, - entropy_lb


def loss_enc(real_v):
    logp_h_given_x, real_h = enc.log_conditional_prob(real_v)
    positve_phase = tf.reduce_mean(rbm.energy(real_h, real_v), axis=0)
    entropy_term = - tf.reduce_mean(logp_h_given_x, axis=0)
    return positve_phase - entropy_term, positve_phase, - entropy_term


lr = tfe.Variable(lr_rbm)
if opt_type is 'mom':
    optimizer_rbm = tf.train.MomentumOptimizer(lr, momentum=0.9)
elif opt_type is 'adam':
    optimizer_rbm = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1_rbm)
optimizer_dec = tf.train.AdamOptimizer(learning_rate=lr_dec, beta1=beta_1_dec)
optimizer_enc = tf.train.AdamOptimizer(learning_rate=lr_enc, beta1=beta_1_enc)


'''
evaluate loop
'''
def evaluate_ll(ds):
    log_z = rbm.log_z_summing_h()
    ds = ds.batch(evaluation_batch_size)
    ds_iter = tfe.Iterator(ds)
    ll = []
    for batch in ds_iter:
        ll.append(tf.reduce_mean(rbm.logprob_vis(batch, log_z)).numpy())
    return log_z.numpy(), np.mean(ll)

def evaluate_ais_ll(ds, data_train):
    _, ais_log_z, _ = estimate_log_partition_function(data_train, rbm)
    ds = ds.batch(evaluation_batch_size)
    ds_iter = tfe.Iterator(ds)
    ais_ll = []
    for batch in ds_iter:
        ais_ll.append(tf.reduce_mean(likelihood_ais(batch, rbm, ais_log_z)).numpy())
    return ais_log_z.numpy(), np.mean(ais_ll)


def evaluate_ll_train(dataset_iter, train_dataset, log_z):
    ais_ll = []
    for i in xrange(20):
        try:
            x_batch = dataset_iter.next()
        except StopIteration:
            dataset_iter = tfe.Iterator(train_dataset)
            x_batch = dataset_iter.next()
        ais_ll.append(tf.reduce_mean(likelihood_ais(x_batch, rbm, log_z)).numpy())
    return np.mean(ais_ll)


def test_lb():
    en, lb = [], []
    for i in xrange(20):
        logprob_v_and_h, _, _ = dec.log_prob(500)
        en.append(- tf.reduce_mean(logprob_v_and_h, axis=0))

        logprob_v_and_h_and_z, fake_z, fake_h, _ = dec.log_prob_all(500)
        logprob_z_given_h = enc_h2z.log_conditional_prob_evaluate(fake_h, fake_z)
        lb.append(tf.reduce_mean(logprob_z_given_h - logprob_v_and_h_and_z, axis=0))
    return np.mean(en), np.mean(lb)


'''
train loop
'''
average_loss_rbm = 0.
average_loss_dec = 0.
average_loss_enc = 0.
average_energy_pos = 0.
average_energy_neg = 0.
average_ent_enc = 0.
average_ent_dec = 0.
start = time.time()
ais_ll_list, ais_log_z_list = [], []
energy_pos_list, energy_neg_list, ent_enc_list, ent_dec_list = [], [], [], []
rbm_loss_list, enc_loss_list, dec_loss_list, loss_list = [], [], [], []
en, lb = [], []
x_batch = dataset_iter.next()
utils.save_images(x_batch.numpy().reshape((batch_size,) + pixels), os.path.join(outf, 'data.png'))
for step in range(1, num_steps + 1):

    # after one epoch, shuffle and refill the quene
    try:
        x_batch = dataset_iter.next()
    except StopIteration:
        dataset_iter = tfe.Iterator(train_dataset)
        x_batch = dataset_iter.next()

    if opt_type is not 'adam':
        lr.assign(tf.train.polynomial_decay(lr_rbm, step, decay_steps, lr_end, power=0.5))

    # update the variables following gradients info
    with tf.GradientTape() as rbm_tape:
        batch_loss_rbm = loss_rbm(x_batch)
        average_loss_rbm += batch_loss_rbm
    grad_rbm = rbm_tape.gradient(batch_loss_rbm, rbm.params())
    optimizer_rbm.apply_gradients(zip(grad_rbm, rbm.params()))

    for j in xrange(dec_per_rbm):
        with tf.GradientTape() as dec_tape:
            batch_loss_dec, batch_energy_neg, batch_ent_dec = loss_dec()
            average_loss_dec += batch_loss_dec
            average_energy_neg += batch_energy_neg
            average_ent_dec += batch_ent_dec
        grad_dec = dec_tape.gradient(batch_loss_dec, dec.params() + enc_h2z.params())
        optimizer_dec.apply_gradients(zip(grad_dec, dec.params() + enc_h2z.params()))

    for j in xrange(enc_per_rbm):
        # after one epoch, shuffle and refill the quene
        try:
            x_batch_enc = dataset_iter.next()
        except StopIteration:
            dataset_iter = tfe.Iterator(train_dataset)
            x_batch_enc = dataset_iter.next()

        with tf.GradientTape() as enc_tape:
            batch_loss_enc, batch_energy_pos, batch_ent_enc = loss_enc(x_batch_enc)
            average_loss_enc += batch_loss_enc
            average_energy_pos += batch_energy_pos
            average_ent_enc += batch_ent_enc
        grad_enc = enc_tape.gradient(batch_loss_enc, enc.params())
        optimizer_enc.apply_gradients(zip(grad_enc, enc.params()))

    # verbose
    if step % ver_steps == 0:
        if dim_z2h[0] <= 15:
            en_batch, lb_batch = test_lb()
            en.append(en_batch)
            lb.append(lb_batch)
            print ("exact entropy = %08f, lower bound = %08f\n" % (en_batch, lb_batch))
            with open(logfile, 'a') as f:
                f.write("exact entropy = %08f, lower bound = %08f\n" % (en_batch, lb_batch))
        average_loss_rbm /= ver_steps
        average_loss_enc /= (ver_steps * enc_per_rbm)
        average_loss_dec /= (ver_steps * dec_per_rbm)
        average_energy_pos /= (ver_steps * enc_per_rbm)
        average_energy_neg /= (ver_steps * dec_per_rbm)
        average_ent_enc /= (ver_steps * enc_per_rbm)
        average_ent_dec /= (ver_steps * dec_per_rbm)

        if hid_dim <= 15:
            ais_log_z, ais_ll = evaluate_ll(validation_dataset)
        else:
            ais_log_z, ais_ll = evaluate_ais_ll(validation_dataset, x_batch)
        ais_ll_train = evaluate_ll_train(dataset_iter, train_dataset, ais_log_z)
        ais_ll_list.append(ais_ll)
        ais_log_z_list.append(ais_log_z)

        rbm_loss_list.append(average_loss_rbm.numpy())
        enc_loss_list.append(average_loss_enc.numpy())
        dec_loss_list.append(average_loss_dec.numpy())
        energy_pos_list.append(average_energy_pos.numpy())
        energy_neg_list.append(average_energy_neg.numpy())
        ent_enc_list.append(average_ent_enc.numpy())
        ent_dec_list.append(average_ent_dec.numpy())
        loss_list.append(-(average_loss_enc - average_loss_dec).numpy())
        end = time.time()
        print("Step: %05d, rbm loss = %08f, dec loss = %08f, ais log Z = %08f, enc loss = %08f, ais ll = %08f, ais train = %08f, time = %08f\n" %
              (step, average_loss_rbm, average_loss_dec, ais_log_z, average_loss_enc, ais_ll, ais_ll_train, end - start))
        with open(logfile, 'a') as f:
            f.write("Step: %05d, rbm loss = %08f, dec loss = %08f, ais log Z = %08f, enc loss = %08f, ais ll = %08f, ais train = %08f, time = %08f\n" %
                    (step, average_loss_rbm, average_loss_dec, ais_log_z, average_loss_enc, ais_ll, ais_ll_train, end - start))
        average_loss_rbm = 0.
        average_loss_dec = 0.
        average_loss_enc = 0.
        average_energy_pos = 0.
        average_energy_neg = 0.
        average_ent_enc = 0.
        average_ent_dec = 0.
        start = time.time()

    try:
        # visualization
        if step % vis_steps == 0:
            dec_samples = dec.get_v_mean(num_samples)
            utils.save_images(dec_samples.reshape((num_samples,) + pixels), os.path.join(outf, 'dec_{}_{}.png'.format(method, step)))
            samples = rbm.get_independent_means(num_samples)
            utils.save_images(samples.reshape((num_samples,) + pixels), os.path.join(outf, 'rbm_{}_{}.png'.format(method, step)))
            chain_samples = rbm.get_samples_single_chain(num_samples)
            utils.save_images(
                chain_samples.reshape(
                    (num_samples,
                     ) +
                    pixels),
                os.path.join(
                    outf,
                    'rbm_single_chain_{}_{}.png'.format(
                        method,
                        step)),
                size=(
                    num_samples /
                    adjacent_samples,
                    adjacent_samples))
            # plotting filters
            w = rbm.w.numpy()
            weights = np.transpose(w).reshape((hid_dim,) + pixels)
            utils.save_images(weights[:100], os.path.join(outf, 'filters_{}_{}.png'.format(method, step)), normalize=True)

            # plotting curves
            to_plot_list = [loss_list, ais_ll_list, dec_loss_list, ent_dec_list, energy_pos_list, energy_neg_list, rbm_loss_list, ais_log_z_list]
            fig = plt.figure(figsize=(6, 5))
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, loss_list, 'r-')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, ais_ll_list, 'k-')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, dec_loss_list, 'b-')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, ent_dec_list, 'b:')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, energy_pos_list, 'g-')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, energy_neg_list, 'g:')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, rbm_loss_list, 'c-')
            plt.plot((np.arange(step / ver_steps) + 1) * ver_steps, ais_log_z_list, 'y-')
            legend = plt.legend(('neg_loss', 'ais LL', 'dec loss', 'dec entropy', 'pos free energy', 'neg free energy', 'rbm loss', 'ais log Z'))
            plt.grid(True)
            ylim_min, ylim_max = -200., 100.
            while True:
                should_stop = False
                for plot_list in to_plot_list:
                    if np.sum(np.array(plot_list) >= ylim_min) <= 0.9 * len(plot_list):
                        should_stop = True
                        break
                if should_stop:
                    ylim_min -= 5.
                    break
                ylim_min += 5.
            while True:
                should_stop = False
                for plot_list in to_plot_list:
                    if np.sum(np.array(plot_list) <= ylim_max) <= 0.9 * len(plot_list):
                        should_stop = True
                        break
                if should_stop:
                    ylim_max += 5.
                    break
                ylim_max -= 5.
            plt.ylim((ylim_min, ylim_max))
            plt.tight_layout()
            plt.savefig(os.path.join(outf, 'curve.png'), bbox_inches='tight')
            plt.close(fig)

        # save model
        # if step == num_steps:
            f = open(os.path.join(outf, 'figvalue' + str(step) + '.pkl'), 'wb')
            cPickle.dump(to_plot_list, f)
            if dim_z2h[0] <= 15:
                np.savez(os.path.join(outf, 'lists_of_entropy_bound.npz'), en=en, lb=lb)
            saver = tf.contrib.eager.Saver(rbm.params() + dec.params() + enc_h2z.params() + enc.params())
            save_path = saver.save(os.path.join(outf, 'model.ckpt'))
            print("Model saved in path: %s" % save_path)
    except IOError as xxx_todo_changeme:
        (errno, strerror) = xxx_todo_changeme.args
        print "I/O error({0}): {1}".format(errno, strerror)
    except Exception as e:
        raise e
