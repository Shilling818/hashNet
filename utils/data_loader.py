import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import scipy.io as sio
import h5py


def loadData(root_path, batch_size, num_workers, arch):
    print('Load Train Dataset....')
    train_dataloader = DataLoader(
        antennaDataset(os.path.join(root_path, 'train.mat'), arch),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    print('Load Valid Dataset....')
    valid_dataloader = DataLoader(
        antennaDataset(os.path.join(root_path, 'valid.mat'), arch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print('Load Retrieval Dataset....')
    retrieval_dataloader = DataLoader(
        antennaDataset(os.path.join(root_path, 'valid.mat'), arch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print('Load simMatrix....')
    with h5py.File(os.path.join(root_path, 'post/adjacent.h5'), 'r') as f:
        simMatrix = f['R1'][:]
    f.close()
    simMatrix = np.array(simMatrix, dtype=np.float32)
    # simMatrix = sio.loadmat(os.path.join(root_path, 'adjacent.mat'))

    print('Data Loaded!')

    return train_dataloader, valid_dataloader, retrieval_dataloader, simMatrix


class antennaDataset(Dataset):
    def __init__(self, path, arch):
        path_split = os.path.splitext(path)
        data = None
        if path_split[1] == '.npy':
            data = np.load(path)
        elif path_split[1] == '.mat':
            data = sio.loadmat(path)

        self.data = data['data']
        if arch == 'conv':
            # NHC --> NCH
            self.data = self.data[:, np.newaxis, :]
            # self.data = np.transpose(self.data, (0, 2, 1))
        elif arch == 'fc':
            self.data = np.squeeze(self.data)
        # float64
        # self.data = np.array(self.data, dtype=float)
        self.data = np.array(self.data, dtype=np.float32)
        self.index = np.squeeze(data['index'])
        self.code = data['code']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        index = self.index[item]

        return data, index, item

    def get_indexs(self):
        return self.index

    def get_size(self):
        return len(self.data[0])

    def get_codes(self):
        return self.code

    def get_data(self):
        return self.data
