import os
import sys
import shutil
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import dataset
import utils
from RBM import RBM
from AIS import likelihood_ais, estimate_log_partition_function

sys.setrecursionlimit(100000)

# set eager API
tfe.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


'''
parameters and data
'''
# algorithm
method = 'cd'  # 'cd', 'pcd'
assert method in ['cd', 'pcd']
settings = {
    'adult-1': (123, 50, 'adult', 'sgd'),
    'adult-2': (123, 50, 'adult', 'adam'),
    'adult-3': (123, 50, 'adult', 'same'),
    'connect4-1': (126, 50, 'connect4', 'sgd'),
    'connect4-2': (126, 50, 'connect4', 'adam'),
    'connect4-3': (126, 50, 'connect4', 'same'),
    'dna-1': (180, 50, 'dna', 'sgd'),
    'dna-2': (180, 50, 'dna', 'adam'),
    'dna-3': (180, 50, 'dna', 'same'),
    'mushrooms-1': (112, 50, 'mushrooms', 'sgd'),
    'mushrooms-2': (112, 50, 'mushrooms', 'adam'),
    'mushrooms-3': (112, 50, 'mushrooms', 'same'),
    'nips-1': (500, 200, 'nips', 'sgd'),
    'nips-2': (500, 200, 'nips', 'adam'),
    'nips-3': (500, 200, 'nips', 'same'),
    'ocr_letters-1': (128, 50, 'ocr_letters', 'sgd'),
    'ocr_letters-2': (128, 50, 'ocr_letters', 'adam'),
    'ocr_letters-3': (128, 50, 'ocr_letters', 'same'),
    'rcv1-1': (150, 50, 'rcv1', 'sgd'),
    'rcv1-2': (150, 50, 'rcv1', 'adam'),
    'rcv1-3': (150, 50, 'rcv1', 'same'),
    'web-1': (300, 100, 'web', 'sgd'),
    'web-2': (300, 100, 'web', 'adam'),
    'web-3': (300, 100, 'web', 'same'),
}
setting = settings['web-1']

optimizer_type = setting[3]
train_mc_steps = 1
# data and model
vis_dim = setting[0]
hid_dim = setting[1]
shuffle_buffer = 10000
# optimization
if optimizer_type is 'adam':
    learning_rate = 1e-3
    anneal_steps = 10000
elif optimizer_type is 'same':
    learning_rate = 0.0003
    anneal_steps = 100000000
elif optimizer_type is 'sgd':
    learning_rate = 0.05
    anneal_steps = 5000
else:
    print 'unknown optimizer type'
    exit(0)
batch_size = 100
num_steps = 200000
ver_steps = 1000
# evaluation
evaluation_batch_size = 500
num_samples = 100
# log
filename_script = os.path.basename(os.path.realpath(__file__))
outf = os.path.join("result", os.path.splitext(filename_script)[0])
outf += '.'
outf += setting[2]
outf += '.'
outf += method
outf += str(train_mc_steps)
outf += '.rbm.'
outf += str([hid_dim, vis_dim])
outf += '.'
outf += str(optimizer_type)
outf += '.'
outf += str(learning_rate)
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
train_dataset, validation_dataset, test_dataset = dataset.uci_binary_dataset(filename=setting[2])
train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
dataset_iter = tfe.Iterator(train_dataset)


'''
model and loss
'''
rbm = RBM(vis_dim=vis_dim, hid_dim=hid_dim)
if method is 'cd':
    def loss_fn(v_0):
        v_n = rbm.cd_step(v_0, train_mc_steps)
        return rbm.cd_loss(v_0, v_n)
elif method is 'pcd':
    pass
else:
    print 'unknown method'
    exit()

learning_rate_tf = tfe.Variable(tf.ones([]) * learning_rate, dtype=tf.float32, name='learning_rate')
if optimizer_type is 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf)
if optimizer_type is 'same':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf)
if optimizer_type is 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_tf)
grad = tfe.implicit_gradients(loss_fn)


'''
evaluate loop
'''


def evaluate_ais_ll(ds, data_train):
    _, ais_log_z, _ = estimate_log_partition_function(data_train, rbm)
    ds = ds.batch(evaluation_batch_size)
    ds_iter = tfe.Iterator(ds)
    ais_ll = []
    for batch in ds_iter:
        ais_ll.append(tf.reduce_mean(likelihood_ais(batch, rbm, ais_log_z)).numpy())
    return ais_log_z.numpy(), np.mean(ais_ll)


'''
train loop
'''
saver = tf.contrib.eager.Saver(rbm.params())
average_loss = 0.
start = time.time()
ais_ll_list, ais_log_z_list = [], []
best_valid_ll, best_test_ll = -np.inf, -np.inf
for step in range(1, num_steps + 1):

    # after one epoch, shuffle and refill the quene
    try:
        x_batch = dataset_iter.next()
    except StopIteration:
        dataset_iter = tfe.Iterator(train_dataset)
        x_batch = dataset_iter.next()

    # anneal learning rate if necessary
    if step % anneal_steps == 0:
        learning_rate_tf.assign(learning_rate / (step / anneal_steps * 10))
        print('anneal learning rate to %08f\n' % (learning_rate_tf.numpy()))
        with open(logfile, 'a') as f:
            f.write('anneal learning rate to %08f\n' % (learning_rate_tf.numpy()))

    # update the variables following gradients info
    batch_loss = loss_fn(x_batch)
    average_loss += batch_loss
    optimizer.apply_gradients(grad(x_batch))

    # verbose
    if step % ver_steps == 0:
        average_loss /= ver_steps
        ais_log_z, ais_ll = evaluate_ais_ll(validation_dataset, x_batch)
        ais_ll_list.append(ais_ll)
        ais_log_z_list.append(ais_log_z)
        ais_log_z_test, ais_ll_test = evaluate_ais_ll(test_dataset, x_batch)
        if ais_ll > best_valid_ll:
            best_valid_ll = ais_ll
            best_test_ll = ais_ll_test
            # save model
            np.savez(os.path.join(outf, 'lists_of_results.npz'), ais_ll_list=ais_ll_list, ais_log_z_list=ais_log_z_list)
            save_path = saver.save(os.path.join(outf, 'model.ckpt'))
            print("Model saved in path: %s" % save_path)
        end = time.time()
        print("Step: %05d, train loss = %08f, ais ll = %08f, ais test = %08f, best test = %08f, time = %08f\n" %
              (step, average_loss, ais_ll, ais_ll_test, best_test_ll, end - start))
        with open(logfile, 'a') as f:
            f.write("Step: %05d, train loss = %08f, ais ll = %08f, ais test = %08f, best test = %08f, time = %08f\n" %
                    (step, average_loss, ais_ll, ais_ll_test, best_test_ll, end - start))
        average_loss = 0.
        start = time.time()
