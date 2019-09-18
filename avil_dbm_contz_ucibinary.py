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
from DBM import DBM
from AIS import estimate_log_partition_function_dbm, lowerbound_dbm
from VAR_DEC import VAEDEC_ZHV2
from VAR_ENC import ENC_CONT, SBNENC
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
    'adult-1': (123, 50, 25, [], [], [], 'sigmoid', 'adult', 500),
    'connect4-1': (126, 50, 25, [], [], [], 'sigmoid', 'connect4', 200),
    'digits-1': (64, 50, 25, [], [], [], 'sigmoid', 'digits', 200),
    'dna-1': (180, 50, 25, [], [], [], 'sigmoid', 'dna', 200),
    'mushrooms-1': (112, 50, 25, [], [], [], 'sigmoid', 'mushrooms', 200),
    'nips-1': (500, 200, 50, [], [], [], 'sigmoid', 'nips', 10),
    'ocr_letters-1': (128, 50, 25, [], [], [], 'sigmoid', 'ocr_letters', 200),
    'rcv1-1': (150, 50, 25, [], [], [], 'sigmoid', 'rcv1', 200),
    'web-1': (300, 100, 50, [], [], [], 'sigmoid', 'web', 200),
    'adult-2': (123, 50, 15, [25], [], [], 'sigmoid', 'adult', 500),
    'connect4-2': (126, 50, 15, [25], [], [], 'sigmoid', 'connect4', 500),
    'digits-2': (64, 50, 15, [25], [], [], 'sigmoid', 'digits', 500),
    'dna-2': (180, 50, 15, [25], [], [], 'sigmoid', 'dna', 500),
    'mushrooms-2': (112, 50, 15, [25], [], [], 'sigmoid', 'mushrooms', 500),
    'nips-2': (500, 200, 50, [100], [], [], 'sigmoid', 'nips', 10),
    'ocr_letters-2': (128, 50, 15, [50], [], [], 'sigmoid', 'ocr_letters', 500),
    'rcv1-2': (150, 50, 15, [25], [], [], 'sigmoid', 'rcv1', 500),
    'web-2': (300, 100, 25, [50], [], [], 'sigmoid', 'web', 500),
}
setting = settings['web-2']

vis_dim = setting[0]
hid_dim2 = setting[1]
hid_dim1 = setting[1]
z_dim = setting[2]

# dim_zh2 = [z_dim, hid_dim2]
dim_zh2 = [z_dim] + setting[3] + [hid_dim2]
dim_h2h1 = [hid_dim2, hid_dim1]
dim_h1v = [hid_dim1, vis_dim]

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
ver_steps = setting[8]
# evaluation
evaluation_batch_size = 500
num_samples = 100
vis_steps = 5000
# log
filename_script = os.path.basename(os.path.realpath(__file__))
outf = os.path.join("result", os.path.splitext(filename_script)[0])
outf += '.'
outf += setting[7]
outf += '.'
outf += opt_type
outf += '.bs.'
outf += str(batch_size)
outf += '.rbm.'
outf += str([hid_dim2, hid_dim1, vis_dim])
outf += '.enc.'
outf += str(enc_per_rbm)
outf += '.dec.'
outf += str(dec_per_rbm)
outf += '.'
outf += str(dim_zh2)
outf += '.'
outf += str(dim_h2h1)
outf += '.'
outf += str(dim_h1v)
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
train_dataset, validation_dataset, test_dataset = dataset.uci_binary_dataset(filename=setting[7])
train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
dataset_iter = tfe.Iterator(train_dataset)


'''
model and loss
'''
dbm = DBM(vis_dim=vis_dim, hid_dim1=hid_dim1, hid_dim2=hid_dim2)
dec = VAEDEC_ZHV2(dim_zh2, dim_h2h1, dim_h1v, activation_fn=non, temp=.1, hard=False)
enc = SBNENC(dim_h1v[::-1], dim_h2h1[::-1], activation_fn=non, temp=.1, hard=False)
enc_h2z = ENC_CONT(dim_zh2[::-1], name='z2h', activation_fn=non)

def loss_rbm(real_v):
    _, real_h1, real_h2 = enc.log_conditional_prob(real_v)
    positve_phase = tf.reduce_mean(dbm.energy(real_h2, real_h1, real_v), axis=0)
    _, _, fake_h2, fake_h1, fake_v = dec.log_prob_all(batch_size)
    negative_phase = tf.reduce_mean(dbm.energy(fake_h2, fake_h1, fake_v), axis=0)
    return positve_phase - negative_phase

def loss_dec():
    logprob_v_and_h_and_z, fake_z, fake_h2, fake_h1, fake_v = dec.log_prob_all(batch_size)
    negative_phase = tf.reduce_mean(dbm.energy(fake_h2, fake_h1, fake_v), axis=0)
    logprob_z_given_h = enc_h2z.log_conditional_prob_evaluate(fake_h2, fake_z)
    entropy_lb = tf.reduce_mean(logprob_z_given_h - logprob_v_and_h_and_z, axis=0)
    return negative_phase - entropy_lb, negative_phase, - entropy_lb

def loss_enc(real_v):
    logp_h_given_x, real_h1, real_h2 = enc.log_conditional_prob(real_v)
    positve_phase = tf.reduce_mean(dbm.energy(real_h2, real_h1, real_v), axis=0)
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


def evaluate_ais_ll(ds, data_train):
    _, ais_log_z, _ = estimate_log_partition_function_dbm(dbm)
    ds = ds.batch(evaluation_batch_size)
    ds_iter = tfe.Iterator(ds)
    ais_ll = []
    for batch in ds_iter:
        ais_ll.append(tf.reduce_mean(lowerbound_dbm(batch, dbm, enc, ais_log_z)).numpy())
    return ais_log_z.numpy(), np.mean(ais_ll)

def evaluate_ais_ll_with_logz(ds, ais_log_z):
    ds = ds.batch(evaluation_batch_size)
    ds_iter = tfe.Iterator(ds)
    ais_ll = []
    for batch in ds_iter:
        ais_ll.append(tf.reduce_mean(lowerbound_dbm(batch, dbm, enc, ais_log_z)).numpy())
    return np.mean(ais_ll)


def evaluate_ll_train(dataset_iter, train_dataset, log_z):
    ais_ll = []
    for i in xrange(20):
        try:
            x_batch = dataset_iter.next()
        except StopIteration:
            dataset_iter = tfe.Iterator(train_dataset)
            x_batch = dataset_iter.next()
        ais_ll.append(tf.reduce_mean(lowerbound_dbm(x_batch, dbm, enc, log_z)).numpy())
    return np.mean(ais_ll)


'''
train loop
'''
saver = tf.contrib.eager.Saver(dbm.params() + dec.params() + enc_h2z.params() + enc.params())
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
best_valid_ll, best_test_ll = -np.inf, -np.inf
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
    grad_rbm = rbm_tape.gradient(batch_loss_rbm, dbm.params())
    optimizer_rbm.apply_gradients(zip(grad_rbm, dbm.params()))

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
        average_loss_rbm /= ver_steps
        average_loss_enc /= (ver_steps * enc_per_rbm)
        average_loss_dec /= (ver_steps * dec_per_rbm)
        average_energy_pos /= (ver_steps * enc_per_rbm)
        average_energy_neg /= (ver_steps * dec_per_rbm)
        average_ent_enc /= (ver_steps * enc_per_rbm)
        average_ent_dec /= (ver_steps * dec_per_rbm)

        ais_log_z, ais_ll = evaluate_ais_ll(validation_dataset, x_batch)
        # speed up by reusing ais_log_z
        ais_ll_test = evaluate_ais_ll_with_logz(test_dataset, ais_log_z)
        if ais_ll > best_valid_ll:
            best_valid_ll = ais_ll
            best_test_ll = ais_ll_test
            # save model
            try:
                save_path = saver.save(os.path.join(outf, 'model.ckpt'))
                print("Model saved in path: %s" % save_path)
            except IOError as xxx_todo_changeme:
                (errno, strerror) = xxx_todo_changeme.args
                print "I/O error({0}): {1}".format(errno, strerror)
            except Exception as e:
                raise e
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
        print("Step: %05d, rbm loss = %08f, dec loss = %08f, ais log Z = %08f, enc loss = %08f, "
              "ais ll = %08f, ais train = %08f, ais test = %08f, best test = %08f, time = %08f\n" %
              (step, average_loss_rbm, average_loss_dec, ais_log_z, average_loss_enc, ais_ll, ais_ll_train, ais_ll_test, best_test_ll, end - start))
        try:
            with open(logfile, 'a') as f:
                f.write("Step: %05d, rbm loss = %08f, dec loss = %08f, ais log Z = %08f, enc loss = %08f, "
                        "ais ll = %08f, ais train = %08f, ais test = %08f, best test = %08f, time = %08f\n" %
                        (step, average_loss_rbm, average_loss_dec, ais_log_z, average_loss_enc, ais_ll, ais_ll_train, ais_ll_test, best_test_ll, end - start))
        except BaseException:
            pass
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
            np.savez(os.path.join(outf, 'lists_of_results.npz'), ais_ll_list=ais_ll_list, ais_log_z_list=ais_log_z_list)
            f = open(os.path.join(outf, 'figvalue.pkl'), 'wb')
            cPickle.dump(to_plot_list, f)
    except IOError as xxx_todo_changeme:
        (errno, strerror) = xxx_todo_changeme.args
        print "I/O error({0}): {1}".format(errno, strerror)
    except Exception as e:
        raise e
