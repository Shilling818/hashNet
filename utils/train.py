import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
import time
import os

from .model import Network, FCNetwork
from .loss import DHNLoss
from .metric import mean_average_precision, pr_curve


def generateCode(model, dataloader, code_length, device):
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()
    model.train()
    return code


def trainModel(train_dataloader,
               valid_dataloader,
               retrieval_dataloader,
               simMatrix,
               code_length,
               device,
               lr,
               max_iter,
               lamda,
               topk,
               evaluate_interval=10,
               arch='conv',
               load_checkpoints_path=None,
               save_checkpoints_path=None):
    data_length = train_dataloader.dataset.get_size()

    if arch == 'conv':
        nfilters = 32
        model = Network(data_length, code_length, nfilters)
    elif arch == 'fc':
        model = FCNetwork(data_length, code_length)
    else:
        print("No Suitable Network Architecture!")

    if load_checkpoints_path is not None:
        model.load_state_dict(load_checkpoints_path, strict=False)

    model.to(device)

    criterion = DHNLoss(lamda)

    # optimizer = torch.optim.RMSprop(
    #     model.parameters(),
    #     lr=lr,
    #     weight_decay=5e-4,
    # )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     max_iter,
    #     lr / 100,
    # )

    # intialization
    best_map = 0.
    training_time = 0.
    running_loss = 0.

    # training
    print('Train Begin....')
    for iter in range(max_iter):
        tic = time.time()
        for data, index, idn in train_dataloader:
            # load from simMatrix
            index = index.squeeze()
            S = simMatrix[index, :][:, index]
            S = torch.from_numpy(S).to(device)

            # batch_size = data.shape[0]
            # S = torch.eye(batch_size).to(device)

            data = data.to(device)

            # S = (target @ target.t() > 0).float()

            output = model(data)
            loss = criterion(output, S)
            running_loss += loss.item()

            del S
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        training_time += time.time() - tic

        if iter % evaluate_interval == evaluate_interval - 1:
            valid_code = generateCode(model, valid_dataloader, code_length, device)
            retrieval_code = generateCode(model, retrieval_dataloader, code_length, device)

            # valid_target = valid_dataloader.dataset.get_onehot_targets()
            # retrieval_target = retrieval_dataloader.dataset.get_onehot_targets()
            valid_index = valid_dataloader.dataset.get_indexs()
            retrieval_index = retrieval_dataloader.dataset.get_indexs()
            retrieval_index = retrieval_index.squeeze()

            part = 16
            dePart = valid_index.shape[0] // part
            mAP_valid = 0.
            P_valid, R_valid = torch.zeros(part * dePart, code_length + 1).to(device), \
                               torch.zeros(part * dePart, code_length + 1).to(device)
            for i in range(dePart):
                valid_part_index = valid_index[(i * part): (i + 1) * part]
                valid_part_index = valid_part_index.squeeze()
                valid_part_code = valid_code[(i * part): (i + 1) * part]
                # load from simMatrix
                valid_simMatrix = simMatrix[valid_part_index, :][:, retrieval_index]
                valid_simMatrix = torch.from_numpy(valid_simMatrix).to(device)

                mAP = mean_average_precision(
                    valid_part_code.to(device),
                    retrieval_code.to(device),
                    valid_simMatrix,
                    device,
                    topk,
                )
                mAP_valid += mAP

                P, R = pr_curve(
                    valid_part_code.to(device),
                    retrieval_code.to(device),
                    valid_simMatrix,
                    device,
                )
                P_valid[(i * part): (i + 1) * part] = P
                R_valid[(i * part): (i + 1) * part] = R

            del valid_simMatrix
            torch.cuda.empty_cache()

            mAP_valid = mAP_valid / part / dePart
            mask = (P_valid > 0).float().sum(dim=0)
            mask = mask + (mask == 0).float() * 0.1
            P_valid = P_valid.sum(dim=0) / mask
            R_valid = R_valid.sum(dim=0) / mask

            # Log
            logger.info('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}]'.format(
                iter + 1,
                max_iter,
                running_loss / evaluate_interval,
                mAP_valid,
                training_time,
            ))
            running_loss = 0.

            # print('Epoch: {}, Loss: {}, mAP: {}, training_time: {}'.format(
            #     (iter + 1), running_loss / (idn + 1), mAP, training_time))

            # checkpoint
            if best_map < mAP_valid:
                best_map = mAP_valid

                checkpoint_best = {
                    'model': model.state_dict(),
                    'valid_code': valid_code.cpu(),
                    'retrieval_code': retrieval_code.cpu(),
                    # 'valid_target': valid_target.cpu(),
                    # 'retrieval_target': retrieval_target.cpu(),
                    'P': P_valid,
                    'R': R_valid,
                    'mAP': best_map,
                }

                torch.save(
                    checkpoint_best,
                    os.path.join(save_checkpoints_path, 'model_best_{}_code_{}_lamda_{}.pt'.format(
                        arch,
                        code_length,
                        lamda))
                )

        checkpoint = {
            'model': model.state_dict(),
            'valid_code': valid_code.cpu(),
            'retrieval_code': retrieval_code.cpu(),
            # 'valid_target': valid_target.cpu(),
            # 'retrieval_target': retrieval_target.cpu(),
            'P': P_valid,
            'R': R_valid,
            'mAP': mAP_valid,
        }

        torch.save(
            checkpoint,
            os.path.join(save_checkpoints_path, 'model_{}_code_{}_lamda_{}.pt'.format(
                arch,
                code_length,
                lamda))
        )

