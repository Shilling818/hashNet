import h5py
import numpy as np
import scipy.io as sio
import os
import sys


if __name__ == '__main__':
    print(os.getcwd())
    root_path = '../data/pro'
    save_path = '../data/post'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # change -1 to 1
    # files = os.scandir(root_path)
    # for file in files:
    #     file = file.name
    #     if file != '0_995.mat':
    #         continue
    #     print('Process %s ....' % file)
    #     with h5py.File(os.path.join(root_path, file), 'r') as f:
    #         simMatrix = f['R1'][:]
    #     f.close()
    #     simMatrix = np.array(simMatrix)
    #     simMatrix = abs(simMatrix)
    #     diag_maxtrix = np.identity(simMatrix.shape[0])
    #     simMatrix = simMatrix + diag_maxtrix
    #     # {-1, 1}
    #     # simMatrix[simMatrix == 0] = -1
    #     sum_column = simMatrix.sum(axis=0)
    #     sum_row = simMatrix.sum(axis=1)
    #
    #     filename, _ = os.path.splitext(file)
    #
    #     f = h5py.File(os.path.join(save_path, filename + '_adjacent.h5'), 'w')
    #     f.create_dataset('R1', data=simMatrix)
    #     f.close()
    #     sio.savemat(os.path.join(save_path, filename + '_sum.mat'),
    #                 {'sum_column': sum_column, 'sum_row': sum_row})

    # make the diagonal matrix with 1
    data_shape = 65536
    simMatrix = np.identity(data_shape)
    f = h5py.File(os.path.join(save_path, 'adjacent.h5'), 'w')
    f.create_dataset('R1', data=simMatrix)
    f.close()

