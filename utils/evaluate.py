import torch
from torch.utils.data.dataloader import DataLoader
from .model import Network, FCNetwork
from .data_loader import antennaDataset
import scipy.io as sio
import os
import numpy as np


def hammingDistance(code1, code2):
    # code1 = code1.unsqueeze(1)
    # code2 = code2.unsqueeze(0)
    # code1 = code1.bool()
    # code2 = code2.bool()
    # code_bool = (code1 ^ code2).float()
    # return code_bool.sum(axis=-1)
    return 0.5 * (code2.shape[1] - code1 @ code2.t())


def generateHashCode(model, dataloader, code_length, device):
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        hash_code = torch.zeros([N, code_length])

        for data, _, idx in dataloader:
            data = data.to(device)
            hash_code_tmp = model(data)
            hash_code[idx, :] = hash_code_tmp.sign().cpu()
    return hash_code


def match(test_path, retrieval_path, code_length, model_path, device, batch_size, num_workers, arch):
    test_dataloader, retrieval_dataloader = loadTestData(test_path, retrieval_path, batch_size, num_workers, arch)

    data_length = test_dataloader.dataset.get_size()

    if arch == 'conv':
        nfilters = 32
        model = Network(data_length, code_length, nfilters)
    elif arch == 'fc':
        model = FCNetwork(data_length, code_length)
    else:
        print("No Suitable Network Architecture!")

    mAP = torch.load(model_path)['mAP']
    print('mAP: ', mAP)

    model_parament = torch.load(model_path)['model']
    model.load_state_dict(model_parament, strict=True)
    model.eval()
    for key, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    retrieval_hash_code = generateHashCode(model, retrieval_dataloader, code_length, device)
    test_hash_code = generateHashCode(model, test_dataloader, code_length, device)

    num_test = test_hash_code.shape[0]

    hamming_dist = hammingDistance(test_hash_code, retrieval_hash_code)
    index_hamming_dist_min = hamming_dist.argmin(dim=-1)

    test_data = test_dataloader.dataset.get_data()
    retrieval_data = retrieval_dataloader.dataset.get_data()

    # index_hamming_dist_min_real = np.zeros(num_test, dtype=int)
    # for i in range(num_test):
    #     hamming_dist_min_i = hamming_dist[i, index_hamming_dist_min[i]]
    #     a = hamming_dist[i, :] == hamming_dist_min_i
    #     a_index = np.argwhere(a.numpy() == 1).squeeze()
    #     diss = np.zeros(len(a_index))
    #     for j in range(len(a_index)):
    #         diss[j] = np.abs((test_data[i, :] - retrieval_data[a_index[j], :])).sum()
    #     index_hamming_dist_min_real[i] = a_index[diss.argmin()]

    # index_hamming_dist_min_real = np.zeros(num_test, dtype=int)
    # for i in range(num_test):
    #     hamming_dist_min_i = hamming_dist[i, index_hamming_dist_min[i]]
    #     a = hamming_dist[i, :] == hamming_dist_min_i
    #     a_index = np.argwhere(a.numpy() == 1).squeeze()
    #     retrieval_data_tmp = retrieval_data[a_index, :]
    #     diss = np.abs(test_data[i, :] - retrieval_data_tmp).sum(axis=-1)
    #     print(diss.argmin(), a_index.shape)
    #     index_hamming_dist_min_real[i] = a_index[diss.argmin()]

    # solve the problem of polykeys
    hamming_dist = hamming_dist.numpy()
    hamming_dist_min = np.amin(hamming_dist, axis=-1)
    hamming_dist_diff = (hamming_dist - hamming_dist_min[:, np.newaxis]) == 0
    index_hamming_dist_min = np.argwhere(hamming_dist_diff)
    print('num: ', index_hamming_dist_min.shape[0])
    num_hamming_dist_min = np.insert(np.sum(hamming_dist_diff, axis=-1), 0, 0)
    diss = np.abs(test_data[index_hamming_dist_min[:, 0], :] - retrieval_data[index_hamming_dist_min[:, 1], :]).sum(axis=-1)
    index_hamming_dist_min_real = np.zeros(num_test, dtype=int)
    count = 0
    for i in range(num_test):
        count = num_hamming_dist_min[i] + count
        index = np.argmin(diss[count: (count + num_hamming_dist_min[i + 1])])
        index_hamming_dist_min_real[i] = index_hamming_dist_min[count + index, 1]

    retrieval_code = retrieval_dataloader.dataset.get_codes()
    test_retri_code = retrieval_code[index_hamming_dist_min_real]

    test_index = test_dataloader.dataset.get_indexs()
    test_retri_data = retrieval_data[index_hamming_dist_min_real]

    sio.savemat('test_results.mat', {'test_data': test_data,
                                     'test_index': test_index,
                                     'retri_index': index_hamming_dist_min_real,
                                     'retri_data': test_retri_data,
                                     'retri_code': test_retri_code,
                                     'test_hash_code': test_hash_code.numpy(),
                                     'retri_hash_code': retrieval_hash_code.numpy()})


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



