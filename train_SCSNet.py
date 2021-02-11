"""

Created on: 2/3/2021 3:10 PM 

@File: train_SCSNet.py
@Author: Xufeng Huang (xufenghuang1228@gmail.com & xfhuang@umich.edu)
@Copy Right: Licensed under the MIT License. 

"""
import torch
import torch.utils.data as DATA
import os
import argparse
import tqdm
import numpy as np
from model import SCSNet
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class SingleChannelDataset(DATA.Dataset):
    def __init__(self, data_path):
        self.dataset = np.load(data_path)

    def __getitem__(self, index):
        inputs_geometry = self.dataset[index, 0:768].astype(np.float32)
        inputs_load = self.dataset[index, 768:770].astype(np.float32)
        outputs_stress = self.dataset[index, 770:].astype(np.float32)

        return inputs_geometry, inputs_load, outputs_stress

    def __len__(self):
        return len(self.dataset[:, 0])


def conf_from_argparse(parser):
    parser.add_argument(
        '--batchSize', type=int, default=512, help='input batch size')
    parser.add_argument(
        '--nWorkers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nEpoch', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--outF', type=str, default='results/trained_models_single/', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='dataset/all_data_s.npy', help="dataset path")

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = conf_from_argparse(parser)
    # print(opt)

    scsnet_data = SingleChannelDataset(opt.dataset)
    dataset_size = len(scsnet_data)
    shuffle_dataset = True

    stress = scsnet_data.dataset[:, 770:]
    stress_mean = np.mean(stress)
    stress_min = np.min(stress)
    stress_max = np.max(stress)
    stress_statistic = [['mean', 'max', 'min'], [stress_mean, stress_min, stress_max]]

    train_num = int(np.floor(100000))
    test_num = int(np.floor(dataset_size - train_num))
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.shuffle(indices)
    train_indices = indices[0:train_num]
    test_indices = indices[train_num:train_num + test_num]

    # Creating data samplers and loaders:
    train_sampler = DATA.SubsetRandomSampler(train_indices)
    test_sampler = DATA.SubsetRandomSampler(test_indices)

    train_loader = DATA.DataLoader(
        scsnet_data,
        batch_size=opt.batchSize,
        sampler=train_sampler,
        num_workers=opt.nWorkers)

    test_loader = DATA.DataLoader(
        scsnet_data,
        batch_size=opt.batchSize,
        sampler=test_sampler,
        num_workers=opt.nWorkers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    blue = lambda x: '\033[94m' + x + '\033[0m'

    scsnet = SCSNet()
    scsnet.initialize()
    scsnet.to(device)
    optimizer = torch.optim.Adam(scsnet.parameters(), lr=1e-03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    results_dir = opt.outF + "%s" % time.strftime("%Y%m%d-%H%M%S")
    logs_dir = results_dir + "/logs/"
    height, width = 24, 32
    resolution = height * width
    min_loss = 100000
    best_epoch = 1

    with SummaryWriter(logs_dir) as writer:
        for epoch in tqdm.tqdm(range(opt.nEpoch)):
            mse_train_losses = []
            mse_test_losses = []
            for i, data_train in enumerate(train_loader, 0):
                geos_train, loads_train, stress_train = data_train
                geos_train = geos_train.view(-1, 1, height, width)
                geos_train, loads_train, stress_train = geos_train.cuda(), loads_train.cuda(), stress_train.cuda()
                optimizer.zero_grad()
                scsnet = scsnet.train()
                pred_train = scsnet(geos_train, loads_train)
                pred_train = pred_train.view(-1, resolution)
                target = stress_train.view(-1, resolution)
                loss = F.mse_loss(pred_train, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                lr = scheduler.get_last_lr()
                mse_train_losses.append(loss.cpu().item())

            if epoch % 100 == 0:
                iTest, data_test = next(enumerate(test_loader, 0))
                geos_test, loads_test, stress_test = data_test
                geos_test = geos_test.view(-1, 1, height, width)
                geos_test, loads_test, stress_test = geos_test.cuda(), loads_test.cuda(), stress_test.cuda()
                scsnet = scsnet.eval()
                with torch.no_grad():
                    pred_val = scsnet(geos_test, loads_test)
                pred_val = pred_val.view(-1, resolution)
                target_val = stress_test.view(-1, resolution)
                loss_val = F.mse_loss(pred_val, target_val)
                print('[epoch %d] %s loss: %f min loss: %f at epoch %d ' %
                      (epoch, blue('val'), loss_val.item(), min_loss, best_epoch))
                if loss_val < min_loss:
                    min_loss = loss_val
                    best_epoch = epoch
                    print("save model")
                    torch.save(scsnet.state_dict(), '%s/val_best_model.pth' % (results_dir))
                mse_test_losses.append(loss_val.cpu().item())
            print('[epoch %d] train loss: %f ' % (epoch, loss.item()))

            loss_dict = {
                'Training loss': np.mean(mse_train_losses),
                'Test loss': np.mean(mse_test_losses),
            }
            lr_dict = {'LearningRate': lr[0]}

            # send results to tensorboard
            writer.add_scalars('Loss', loss_dict, epoch)
            writer.add_scalars('LR', lr_dict, epoch)
