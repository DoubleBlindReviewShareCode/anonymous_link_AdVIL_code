import os
import scipy.io
import urllib
import gzip
import pickle as pkl
import numpy as np
import tensorflow as tf


def toy_dataset(ns, vis_dim, mean):

    def get_dataset(n):
        v = np.asarray(
            [np.random.binomial(1, p=mean) for i in range(n)], dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices(v)

    return get_dataset(ns[0]), get_dataset(ns[1]), get_dataset(ns[2])


def get_patches(x, size):
    x = x.reshape((-1, 28, 28))
    start_r = (28 - size) / 2
    start_c = (28 - size) / 2
    return (x[:, start_r:start_r + size, start_c:start_c + size]).reshape(
        (-1, size * size))


def uci_digits_eval():
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.model_selection import train_test_split

    digits = _load_digits()
    X = np.asarray(digits.data, 'float32')

    fakeX = nudge_dataset(X)
    xmin, xmax = np.min(fakeX, 0), np.max(fakeX, 0)

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=0)
    X_train = nudge_dataset(X_train)
    X_train = (X_train - xmin) / (xmax + 0.0001)  # 0-1 scaling
    X_val = nudge_dataset(X_val)
    X_val = (X_val - xmin) / (xmax + 0.0001)  # 0-1 scaling

    return tf.data.Dataset.from_tensor_slices(
        X_train), tf.data.Dataset.from_tensor_slices(
            X_val), tf.data.Dataset.from_tensor_slices(X_val)


def ocr_letters_eval():
    return uci_binary_dataset(filename='ocr_letters')


def uci_binary_dataset(filename='ocr_letters'):
    import h5py
    datasets = [
        'adult',
        'connect4',
        'digits',
        'dna',
        'mushrooms',
        'nips',
        'ocr_letters',
        'rcv1',
        'web'
    ]
    assert filename in datasets
    if filename == 'digits':
        return uci_digits_eval()
    if filename in ['mushrooms', 'web']:
        filename += '.npz'
        mush_dict = np.load('./data/uci_binary/' + filename)
        X_train = mush_dict['train_data']
        X_valid = mush_dict['valid_data']
        X_test = mush_dict['test_data']
    else:
        filename += '.h5'
        f = h5py.File('./data/uci_binary/' + filename, 'r')
        X_train = np.array(f['train'], dtype=np.float32)
        X_valid = np.array(f['valid'], dtype=np.float32)
        X_test = np.array(f['test'], dtype=np.float32)
        f.close()
    xmax = np.max(np.concatenate([X_train, X_valid, X_test], axis=0), axis=0)
    xmin = np.min(np.concatenate([X_train, X_valid, X_test], axis=0), axis=0)
    X_train = (X_train - xmin) / (xmax + 0.0001)  # 0-1 scaling
    X_valid = (X_valid - xmin) / (xmax + 0.0001)  # 0-1 scaling
    X_test = (X_test - xmin) / (xmax + 0.0001)  # 0-1 scaling

    return tf.data.Dataset.from_tensor_slices(X_train), tf.data.Dataset.from_tensor_slices(X_valid), tf.data.Dataset.from_tensor_slices(X_test)


def _download_frey_faces(dataset):
    from scipy.io import loadmat
    origin = (
        'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset + '.mat')
    matdata = loadmat(dataset)
    f = gzip.open(dataset + '.pkl.gz', 'w')
    pkl.dump([matdata['ff'].T], f)


def frey_faces(
        dataset='./data/frey_faces/frey_faces',
        normalize=True,
        standardize=True,
        dequantify=True):
    datasetfolder = os.path.dirname(dataset + '.pkl.gz')
    if not os.path.isfile(dataset + '.pkl.gz'):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_frey_faces(dataset)

    if not os.path.isfile(datasetfolder + '/fixed_split.pkl'):
        urllib.urlretrieve('https://raw.githubusercontent.com/casperkaae/'
                           'extra_parmesan/master/data_splits/'
                           'frey_faces_fixed_split.pkl',
                           datasetfolder + '/fixed_split.pkl')

    f = gzip.open(dataset + '.pkl.gz', 'rb')
    # data = pkl.load(f)[0].reshape(-1, 28, 20).astype('float32')
    data = pkl.load(f)[0].reshape(-1, 560).astype('float32')
    f.close()
    if dequantify:
        data = data + np.random.uniform(0, 1, size=data.shape).astype('float32')
    if normalize:
        normalizer = data.max().astype('float32')
        data = data / normalizer
    if standardize:
        x_mean = np.mean(data, axis=0, keepdims=True)
        x_std = np.std(data, axis=0, keepdims=True) + 1e-3
        # print x_mean, x_std
        data = (data - x_mean) / x_std
        return tf.data.Dataset.from_tensor_slices(data), x_mean, x_std
    return tf.data.Dataset.from_tensor_slices(data)


def mnist_real_dataset():
    pass


def mnist_real_patch_dataset(size=4, standardize=False):
    train, validation, test = _load_mnist_real()
    train = get_patches(train, size=size)
    validation = get_patches(validation, size=size)
    test = get_patches(test, size=size)
    if standardize:
        x_mean = np.mean(train, axis=0, keepdims=True)
        x_std = np.std(train, axis=0, keepdims=True) + 1e-3
        # print x_mean, x_std
        train = (train - x_mean) / x_std
        validation = (validation - x_mean) / x_std
        test = (test - x_mean) / x_std
        return tf.data.Dataset.from_tensor_slices(
            train), tf.data.Dataset.from_tensor_slices(
                validation), tf.data.Dataset.from_tensor_slices(test), x_mean, x_std
    return tf.data.Dataset.from_tensor_slices(
        train), tf.data.Dataset.from_tensor_slices(
            validation), tf.data.Dataset.from_tensor_slices(test)


def mnist_binary_dataset_eval():
    train, validation, test = _load_mnist_binary()
    return tf.data.Dataset.from_tensor_slices(
        np.concatenate((train, validation),
                       axis=0)), tf.data.Dataset.from_tensor_slices(
                           test), tf.data.Dataset.from_tensor_slices(test)


def mnist_binary_dataset():
    train, validation, test = _load_mnist_binary()
    return tf.data.Dataset.from_tensor_slices(
        train), tf.data.Dataset.from_tensor_slices(
            validation), tf.data.Dataset.from_tensor_slices(test)


def mnist_binary_patch_dataset(size=4):
    train, validation, test = _load_mnist_binary()
    train = get_patches(train, size=size)
    validation = get_patches(validation, size=size)
    test = get_patches(test, size=size)
    return tf.data.Dataset.from_tensor_slices(
        train), tf.data.Dataset.from_tensor_slices(
            validation), tf.data.Dataset.from_tensor_slices(test)


def _load_mnist_real(for_validation=10000):

    def load_mnist_images_np(imgs_filename):
        with open(imgs_filename, 'rb') as f:
            import struct
            f.seek(4)
            nimages, rows, cols = struct.unpack('>iii', f.read(12))
            dim = rows * cols

            images = np.fromfile(f, dtype=np.dtype(np.ubyte))
            images = (images / 255.0).astype('float32').reshape((nimages, dim))
        return images

    train_data = load_mnist_images_np(
        os.path.join('./data/mnist_real', 'train-images-idx3-ubyte'))
    test_data = load_mnist_images_np(
        os.path.join('./data/mnist_real', 't10k-images-idx3-ubyte'))

    return train_data[for_validation:], train_data[:for_validation], test_data


def _load_mnist_binary():

    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(
            os.path.join('./data/mnist_binary',
                         'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(
            os.path.join('./data/mnist_binary',
                         'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('./data/mnist_binary',
                           'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')

    return train_data, validation_data, test_data


def nudge_dataset(X):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    from scipy.ndimage import convolve
    direction_vectors = [[[0, 1, 0], [0, 0, 0],
                          [0, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 1],
                          [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]]]

    def shift(x, w): return convolve(x.reshape((8, 8)), mode='constant',
                                     weights=w).ravel()
    X = np.concatenate([X] + [
        np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors
    ])
    return X
