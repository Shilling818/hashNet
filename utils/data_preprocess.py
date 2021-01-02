import numpy as np
import scipy.io as sio
import os
import random
import numpy as np
import h5py


def mat2npy(path):
    data = sio.loadmat(path)
    data = np.array(data)
    new_path = path.replace('.mat', '.npy')
    np.save(new_path, data)


def splitDatasetPart(input_size):
    num = input_size
    index = random.sample(list(np.arange(num)), int(num * 0.4))
    index_valid = random.sample(index, int(num * 0.2))
    index_test = list(set(index) ^ set(index_valid))
    index_train = list(set(np.arange(num)) ^ set(index))
    index_valid.sort()
    index_test.sort()
    index_train.sort()
    return index_train, index_valid, index_test


def splitDataset(path):
    seed = 500
    random.seed(seed)
    np.random.seed(seed)

    f = h5py.File(path, 'r')
    code = f['code'][:]
    data = f['E'][:]
    f.close()
    code = np.transpose(code)
    data = np.transpose(data)

    index_train, index_valid, index_test = splitDatasetPart(len(data))

    root_path = os.path.split(path)[0]

    sio.savemat(os.path.join(root_path, 'train.mat'), {'data': data[index_train],
                                                       'code': code[index_train],
                                                       'index': np.array(index_train)})

    sio.savemat(os.path.join(root_path, 'valid.mat'), {'data': data[index_valid],
                                                       'code': code[index_valid],
                                                       'index': np.array(index_valid)})

    sio.savemat(os.path.join(root_path, 'test.mat'), {'data': data[index_test],
                                                      'code': code[index_test],
                                                      'index': np.array(index_test)})

    sio.savemat(os.path.join(root_path, 'retrieval.mat'), {'data': data,
                                                           'code': code,
                                                           'index': np.arange(len(data))})


if __name__ == '__main__':
    path = '../data/measured_data.mat'
    splitDataset(path)






