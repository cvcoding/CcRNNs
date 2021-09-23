import torch
from torchvision import datasets, transforms

import wfdb.io as wfdbio
import scipy.io as spio
import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch
import random


# a = torch.rand(9)
# print('a:\n', a)
#
# random.shuffle(a)
# print('random.shuffle(a):\n', a)
#
# index = [i for i in range(len(a))]
# print('index\n', index)
# random.shuffle(index)
# print('random.shuffle(index):\n', index)
# print('shuffle tensor:\n', a[index])


def data_generator(root, batch_size):
    # train_set = datasets.MNIST(root=root, train=True, download=True,
    #                            transform=transforms.Compose([
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.1307,), (0.3081,))
    #                            ]))
    # test_set = datasets.MNIST(root=root, train=False, download=True,
    #                           transform=transforms.Compose([
    #                               transforms.ToTensor(),
    #                               transforms.Normalize((0.1307,), (0.3081,))
    #                           ]))

    rset = set()
    file_names = []
    counter = 1
    f = open('data/H_data/RECORDS')

    data_train_total = []
    target_train_total = []
    data_test_total = []
    target_test_total = []
    data_valid_total = []
    target_valid_total = []

    for file in f:
        if counter <= 17:
            file_name = 'data/H_data/'+file.strip()
            file_names.append(file_name)
            ann = wfdbio.rdann(file_name, 'atr')
            rpnote = ann.symbol
            rpeak = ann.sample
            print(np.where(rpeak < 0))
            fs = ann.fs
            header = wfdbio.rdheader(file_name)
            # start_time = '{}时{}分'.format(header.base_time.hour, header.base_time.minute)
            rdata = wfdbio.rdrecord(file_name, channels=[0])
            rdata = rdata.p_signal
            dura_hour = np.fix((max(rpeak) - min(rpeak)) / 3600 / fs)
            dura_min = np.fix((((max(rpeak) - min(rpeak)) / fs) % 3600) / 60)
            dura = "{}时{}分".format(int(dura_hour), int(dura_min))
            # print('\n第{:0>2}个文件,文件名为{},采样频率为{:},有效时间为{:},开始时间为{:}'.format(counter, ann.record_name, ann.fs, dura,
            #                                                               start_time))

            rset = rset | set(rpnote)
            seqlen = 120
            rdata_resample = spsig.resample_poly(rdata, 250, fs)
            # rpeak_resample = np.zeros(shape=(1, len(rpeak)), dtype=np.float)
            rpeak_resample = np.around(rpeak * 250.0 / fs)
            rpnote_resample = np.zeros(shape=(1, int(len(rdata_resample))))

            data_train = np.zeros(shape=(int(len(rpeak)), seqlen))
            target_train = rpnote[:int(len(rpeak))]

            rdata_resample = rdata_resample.squeeze()

            type = [0, 0, 0, 0]
            for j in range(len(rpeak)):
                if rpeak_resample[j] <= seqlen / 2:
                    data_train[j, : int(rpeak_resample[j]) + int(seqlen / 2)] = rdata_resample[: int(rpeak_resample[j]) + int(seqlen / 2)]

                if rpeak_resample[j] > seqlen / 2 and rpeak_resample[j]+ int(seqlen / 2) < len(rdata_resample) :
                    data_train[j, :] = rdata_resample[
                                 int(rpeak_resample[j]) - int(seqlen / 2): int(rpeak_resample[j]) + int(seqlen / 2)]

                if rpeak_resample[j]+ int(seqlen / 2) > len(rdata_resample) :
                    data_train[j, : len(rdata_resample) - int(rpeak_resample[j]) + int(seqlen / 2)] = rdata_resample[int(rpeak_resample[j]) - int(seqlen / 2):]

                if rpnote[j] == 'N':  # 1代表窦性节律

                    target_train[j] = 0

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 1
                    type[0] = type[0] + 1

                elif rpnote[j] == 'V':  # 3代表室性节律

                    target_train[j] = 1

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 3
                    type[1] = type[1] + 1
                elif rpnote[j] == 'A' or rpnote[j] == 'J' or rpnote[j] == 'S':  # 5代表室上性节律

                    target_train[j] = 2

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 5
                    type[2] = type[2] + 1
                else:  # 4代表异常、干扰等

                    target_train[j] = 3

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 4
                    type[3] = type[3] + 1

            data_train = torch.tensor(data_train)
            target_train = torch.tensor(target_train)

            data_train_total.append(data_train)
            target_train_total.append(target_train)

        elif counter <= 21:  #19,20,21,22
            file_name = 'data/H_data/' + file.strip()
            file_names.append(file_name)
            ann = wfdbio.rdann(file_name, 'atr')
            rpnote = ann.symbol
            rpeak = ann.sample
            print(np.where(rpeak < 0))
            fs = ann.fs
            header = wfdbio.rdheader(file_name)
            # start_time = '{}时{}分'.format(header.base_time.hour, header.base_time.minute)
            rdata = wfdbio.rdrecord(file_name, channels=[0])
            rdata = rdata.p_signal
            dura_hour = np.fix((max(rpeak) - min(rpeak)) / 3600 / fs)
            dura_min = np.fix((((max(rpeak) - min(rpeak)) / fs) % 3600) / 60)
            # dura = "{}时{}分".format(int(dura_hour), int(dura_min))
            # print('\n第{:0>2}个文件,文件名为{},采样频率为{:},有效时间为{:},开始时间为{:}'.format(counter, ann.record_name, ann.fs, dura,
            #                                                               start_time))

            rset = rset | set(rpnote)
            seqlen = 120
            rdata_resample = spsig.resample_poly(rdata, 250, fs)
            # rpeak_resample = np.zeros(shape=(1, len(rpeak)), dtype=np.float)
            rpeak_resample = np.around(rpeak * 250.0 / fs)
            rpnote_resample = np.zeros(shape=(1, int(len(rdata_resample))))

            data_test = np.zeros(shape=(int(len(rpeak)), seqlen))
            target_test = rpnote[: int(len(rpeak))]

            rdata_resample = rdata_resample.squeeze()

            type = [0, 0, 0, 0]
            for j in range(len(rpeak)):
                if rpeak_resample[j] <= seqlen / 2:
                    data_test[j, : int(rpeak_resample[j]) + int(seqlen / 2)] = rdata_resample[
                                                                                : int(rpeak_resample[j]) + int(
                                                                                    seqlen / 2)]

                if rpeak_resample[j] > seqlen / 2 and rpeak_resample[j] + int(seqlen / 2) < len(rdata_resample):
                    data_test[j, :] = rdata_resample[
                                       int(rpeak_resample[j]) - int(seqlen / 2): int(rpeak_resample[j]) + int(
                                           seqlen / 2)]

                if rpeak_resample[j] + int(seqlen / 2) > len(rdata_resample):
                    data_test[j, : len(rdata_resample) - int(rpeak_resample[j]) + int(seqlen / 2)] = rdata_resample[
                                                                                                      int(
                                                                                                          rpeak_resample[
                                                                                                              j]) - int(
                                                                                                          seqlen / 2):]

                if rpnote[j] == 'N':  # 1代表窦性节律

                    target_test[j] = 0

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 1
                    type[0] = type[0] + 1

                elif rpnote[j] == 'V':  # 3代表室性节律

                    target_test[j] = 1

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 3
                    type[1] = type[1] + 1
                elif rpnote[j] == 'A' or rpnote[j] == 'J' or rpnote[j] == 'S':  # 5代表室上性节律

                    target_test[j] = 2

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 5
                    type[2] = type[2] + 1
                else:  # 4代表异常、干扰等

                    target_test[j] = 3

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 4
                    type[3] = type[3] + 1

            data_test = torch.tensor(data_test)
            target_test = torch.tensor(target_test)

            data_test_total.append(data_test)
            target_test_total.append(target_test)

        elif counter <= 25:   #23,24,25
            file_name = 'data/H_data/' + file.strip()
            file_names.append(file_name)
            ann = wfdbio.rdann(file_name, 'atr')
            rpnote = ann.symbol
            rpeak = ann.sample
            print(np.where(rpeak < 0))
            fs = ann.fs
            header = wfdbio.rdheader(file_name)
            # start_time = '{}时{}分'.format(header.base_time.hour, header.base_time.minute)
            rdata = wfdbio.rdrecord(file_name, channels=[0])
            rdata = rdata.p_signal
            dura_hour = np.fix((max(rpeak) - min(rpeak)) / 3600 / fs)
            dura_min = np.fix((((max(rpeak) - min(rpeak)) / fs) % 3600) / 60)
            # dura = "{}时{}分".format(int(dura_hour), int(dura_min))
            # print('\n第{:0>2}个文件,文件名为{},采样频率为{:},有效时间为{:},开始时间为{:}'.format(counter, ann.record_name, ann.fs, dura,
            #                                                               start_time))

            rset = rset | set(rpnote)
            seqlen = 120
            rdata_resample = spsig.resample_poly(rdata, 250, fs)
            # rpeak_resample = np.zeros(shape=(1, len(rpeak)), dtype=np.float)
            rpeak_resample = np.around(rpeak * 250.0 / fs)
            rpnote_resample = np.zeros(shape=(1, int(len(rdata_resample))))

            data_valid = np.zeros(shape=(int(len(rpeak)), seqlen))
            target_valid = rpnote[: int(len(rpeak))]

            rdata_resample = rdata_resample.squeeze()

            type = [0, 0, 0, 0]
            for j in range(len(rpeak)):
                if rpeak_resample[j] <= seqlen / 2:
                    data_valid[j, : int(rpeak_resample[j]) + int(seqlen / 2)] = rdata_resample[
                                                                                : int(rpeak_resample[j]) + int(
                                                                                    seqlen / 2)]

                if rpeak_resample[j] > seqlen / 2 and rpeak_resample[j] + int(seqlen / 2) < len(rdata_resample):
                    data_valid[j, :] = rdata_resample[
                                       int(rpeak_resample[j]) - int(seqlen / 2): int(rpeak_resample[j]) + int(
                                           seqlen / 2)]

                if rpeak_resample[j] + int(seqlen / 2) > len(rdata_resample):
                    data_valid[j, : len(rdata_resample) - int(rpeak_resample[j]) + int(seqlen / 2)] = rdata_resample[
                                                                                                      int(
                                                                                                          rpeak_resample[
                                                                                                              j]) - int(
                                                                                                          seqlen / 2):]

                if rpnote[j] == 'N':  # 1代表窦性节律

                    target_valid[j] = 0

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 1
                    type[0] = type[0] + 1

                elif rpnote[j] == 'V':  # 3代表室性节律

                    target_valid[j] = 1

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 3
                    type[1] = type[1] + 1
                elif rpnote[j] == 'A' or rpnote[j] == 'J' or rpnote[j] == 'S':  # 5代表室上性节律

                    target_valid[j] = 2

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 5
                    type[2] = type[2] + 1
                else:  # 4代表异常、干扰等

                    target_valid[j] = 3

                    rpnote_resample[0, int(rpeak_resample[j] - 1)] = 4
                    type[3] = type[3] + 1

            data_valid = torch.tensor(data_valid)
            target_valid = torch.tensor(target_valid)

            data_valid_total.append(data_valid)
            target_valid_total.append(target_valid)

        counter += 1
        print('counter=', counter)

    data_train_total = torch.cat(data_train_total, dim=0)
    target_train_total = torch.cat(target_train_total, dim=0)

    # Do shuffle

    # len_total = len(data_train_total)
    index = [i for i in range(len(data_train_total))]
    random.shuffle(index)
    data_train_total = data_train_total[index, :]
    target_train_total = target_train_total[index]


    torch_dataset = Data.TensorDataset(data_train_total, target_train_total)
    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size)

    data_test_total = torch.cat(data_test_total, dim=0)
    target_test_total = torch.cat(target_test_total, dim=0)

    torch_dataset_test = Data.TensorDataset(data_test_total, target_test_total)
    test_loader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=batch_size)


    # data_validation

    data_valid_total = torch.cat(data_valid_total, dim=0)
    target_valid_total = torch.cat(target_valid_total, dim=0)

    torch_dataset_valid = Data.TensorDataset(data_valid_total, target_valid_total)
    valid_loader = torch.utils.data.DataLoader(torch_dataset_valid, batch_size=batch_size)

    return train_loader, test_loader, valid_loader
