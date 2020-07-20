import os
import sys
import pickle
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import json

def train_val_split(train_val, train_label, train_n):
    train_val = np.array(train_val)
    train_label = np.array(train_label, dtype=np.int64)
    train_n = int(train_n / 10)
    train_key = []
    val_key = []

    for i in range(10):
        key = np.where(train_label == i)[0]
        np.random.shuffle(key)
        train_key.extend(key[:train_n])
        val_key.extend(key[train_n:])
    np.random.shuffle(train_key)
    np.random.shuffle(val_key)

    return train_val[train_key], train_label[train_key], train_val[val_key], train_label[val_key]
    # return train_val, train_label, train_val, train_label

def loadData(train):
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    base_folder = 'cifar-10-batches-py'
    if train ==1:
        downloaded_list = train_list
    else:
        downloaded_list = test_list
    data = []
    targets = []
    for file_name, checksum in downloaded_list:
        file_path = os.path.join('./data', base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            targets.extend(entry['labels'])
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    return data, targets

class CIFAR10_sym(torch.utils.data.Dataset):
    def __init__(self, data, targets,
                 transform=None, target_transform=None, noise_rate=0.2):
        self.transform = transform
        self.target_transform = target_transform
        self.ac_data = data
        self.targets_ac = targets#np.zeros(len(targets), dtype=np.int64)
        self.groud_truth = np.zeros(len(targets), dtype=np.int64)
        for i in range(len(targets)):
            self.groud_truth[i]=targets[i]
        self.one_hot_label = torch.zeros((len(self.targets_ac), 10))
        self.ids = list(range(len(self.targets_ac)))
        self.ifupdated = np.zeros(len(self.targets_ac), dtype=np.int64)
        idxes = np.random.permutation(len(targets))
        count = 0
        noise_file = 'symmetric_noise_'+str(noise_rate)+'.npy'
        if os.path.exists(noise_file):
            self.targets_ac = np.load(noise_file)
            for idx, i in enumerate(idxes):
                self.one_hot_label[i][self.targets_ac[i]] = int(1)
        else:
            for idx, i in enumerate(idxes):
                if idx < len(self.targets_ac) * noise_rate:
                    # targets[idxes[i]] -> another category
                    label_sym = np.random.randint(10, dtype=np.int64)
                    # while label_sym == self.targets_ac[i]:
                    #     label_sym = np.random.randint(10, dtype=np.int64)
                    self.targets_ac[i] = label_sym
                    count += 1
                self.one_hot_label[i][self.targets_ac[i]] = int(1)
            print("save noisy labels to %s ..." % noise_file)
            np.save(noise_file, self.targets_ac)
        print("moise number:", count, np.sum(self.targets_ac[:] != self.groud_truth[:]))
        print('init over')
        # self._load_meta()

    def __len__(self):
        return len(self.targets_ac)

    def update_noisy_label(self, remove_index):
        # remove those noisy labels
        self.targets_ac = np.delete(self.targets_ac, remove_index, axis=0)
        self.ac_data = np.delete(self.ac_data, remove_index, axis=0)
        print('corrected num:', len(self.targets_ac))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, idx = self.ac_data[index], self.targets_ac[index], self.ids[index]
        one_label = self.one_hot_label[index]
        gr_truth = self.groud_truth[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, target, idx, one_label, gr_truth

class CIFAR10_val(torch.utils.data.Dataset):
    def __init__(self, data, target,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ac_data = data
        self.targets_ac = target
        # self.ids = list(range(len(self.targets_ac)))
        # self._load_meta()

    def __len__(self):
        return len(self.targets_ac)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.ac_data[index], self.targets_ac[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_asym(CIFAR10_sym):
    def __init__(self, data, target, transform=None, target_transform=None,noise_rate=0.2):
        self.transform = transform
        self.target_transform = target_transform

        self.ac_data = data
        self.targets_ac = target
        self.ids = list(range(len(self.targets_ac)))
        self.one_hot_label = torch.zeros((len(self.targets_ac), 10))
        self.groud_truth = np.zeros(len(self.targets_ac), dtype=np.int64)

        for i in range(len(self.targets_ac)):
            self.groud_truth[i] = self.targets_ac[i]
        dic = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
        # for i in range(10):
        #     indices = np.where(self.targets_ac[:] == i)[0]
        #     np.random.shuffle(indices)
        #     for j, idx in enumerate(indices):
        #         #if np.random.random() < noise_rate:
        #         if j < noise_rate * len(indices):
        #             # truck -> automobile
        #             if i == 9:
        #                 self.targets_ac[idx] = 1
        #             # bird -> airplane
        #             elif i == 2:
        #                 self.targets_ac[idx] = 0
        #             # cat -> dog
        #             elif i == 3:
        #                 self.targets_ac[idx] = 5
        #             # dog -> cat
        #             elif i == 5:
        #                 self.targets_ac[idx] = 3
        #             # deer -> horse
        #             elif i == 4:
        #                 self.targets_ac[idx] = 7
        #         self.one_hot_label[idx][self.targets_ac[idx]] = 1
        noise_file = 'asymmetric_noise_' + str(noise_rate)+'.npy'
        if os.path.exists(noise_file):
            self.targets_ac = np.load(noise_file)
            for i in range(len(self.targets_ac)):
                self.one_hot_label[i][self.targets_ac[i]] = int(1)
        else:
            for idx in range(len(self.targets_ac)):
                i = self.targets_ac[idx]
                if idx < noise_rate*len(self.targets_ac):
                #if np.random.random() < noise_rate:
                    # truck -> automobile
                    if i == 9:
                        self.targets_ac[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.targets_ac[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.targets_ac[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.targets_ac[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.targets_ac[idx] = 7
                self.one_hot_label[idx][self.targets_ac[idx]] = 1
            print("save noisy labels to %s ..." % noise_file)
            np.save(noise_file, self.targets_ac)

        print(len(self.targets_ac))
        # self._load_meta()
    def __len__(self):
        return len(self.targets_ac)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, idx = self.ac_data[index], self.targets_ac[index], self.ids[index]
        one_label = self.one_hot_label[index]
        gr_truth = self.groud_truth[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, idx, one_label, gr_truth

def load_data_sym10(noise_rate=0.9):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.492, 0.482, 0.446), (0.247, 0.244, 0.262))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.492, 0.482, 0.446), (0.247, 0.244, 0.262))])

    root = './data'
    data, targets = loadData(1)
    train_data, train_label, val_data, val_label = train_val_split(data, targets, int(0.9*len(targets)))

    trainset = CIFAR10_sym(train_data, train_label,
                            transform=transform_train, noise_rate=noise_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)
    # trainloader_step2 = torch.utils.data.DataLoader(trainset, batch_size=16,
    #                                           shuffle=True, num_workers=0)
    testset1 = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                           transform=transform_test)
    testset = CIFAR10_val(val_data, val_label,
                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)
    testloader1 = torch.utils.data.DataLoader(testset1, batch_size=128,
                                             shuffle=False, num_workers=0)

    return trainloader, testloader1, testloader


def load_data_asym10(noise_rate=0.5):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.492, 0.482, 0.446), (0.247, 0.244, 0.262))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.492, 0.482, 0.446), (0.247, 0.244, 0.262))])

    root = './data'
    data, targets = loadData(1)
    train_data, train_label, val_data, val_label = train_val_split(data, targets, int(0.9 * len(targets)))

    trainset = CIFAR10_asym(train_data, train_label,
                             transform=transform_train, noise_rate=noise_rate)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)
    # trainloader_step2 = torch.utils.data.DataLoader(trainset, batch_size=16,
    #                                           shuffle=True, num_workers=0)
    testset1 = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                            transform=transform_test)
    testset = CIFAR10_val(val_data, val_label,
                          transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)
    testloader1 = torch.utils.data.DataLoader(testset1, batch_size=128,
                                              shuffle=False, num_workers=0)

    return trainloader, testloader, testloader1