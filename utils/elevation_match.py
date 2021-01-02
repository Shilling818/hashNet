import torch
import numpy as np
import scipy.io as sio
from torch.utils.data.dataloader import DataLoader
from .data_loader import antennaDataset


def L1_distance(data1, data2):
    data1 = data1[:, np.newaxis, :]
    data2 = data2[np.newaxis, ...]
    data_diff = np.abs(data1 - data2)
    return np.mean(data_diff, axis=-1)


def match(test_path, retrieval_path, batch_size, num_workers, arch):
    test_dataloader, retrieval_dataloader = loadTestData(test_path, retrieval_path, batch_size, num_workers, arch)

    test_data = test_dataloader.dataset.get_data()
    retrieval_data = retrieval_dataloader.dataset.get_data()
    data_diff = L1_distance(test_data, retrieval_data)
    index_data_diff_min = np.argmin(data_diff, axis=-1)
    test_retri_data = retrieval_data[index_data_diff_min]

    test_code = test_dataloader.dataset.get_codes()
    retrieval_code = retrieval_dataloader.dataset.get_codes()
    test_retri_code = retrieval_code[index_data_diff_min]

    sio.savemat('test_results.mat', {'test_data': test_data,
                                     'test_code': test_code,
                                     'retri_data': test_retri_data,
                                     'retri_code': test_retri_code,})


def loadTestData(test_path, retrieval_path, batch_size, num_workers, arch):
    print('Load Test Dataset....')
    test_dataloader = DataLoader(
        antennaDataset(test_path, arch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    retrieval_dataloader = DataLoader(
        antennaDataset(retrieval_path, arch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print('Test Data Loaded!')

    return test_dataloader, retrieval_dataloader