# -*- coding: utf-8 -*-
import pickle
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

class data_new(Dataset):
    def __init__(self,X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    def __getitem__(self, item):
        X = self.X[item]
        if len(X.shape) == 3:
            image = np.concatenate((X, X, X), axis=0)
            #with torch.no_grad():
            #    new_label = self.y[item]+10
            #X = np.repeat(X[None],3,axis=0).astype(np.float32)
        # else:
        #     X = np.transpose(X, (2,0,1)).astype(np.float32)
            #print('data_new;y[item]:',self.y[item],type(self.y[item]))
        return image, self.y[item]

    def __len__(self):
        return len(self.X)


#　迭代数据类
class data(Dataset):
    def __init__(self,X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    def __getitem__(self, item):
        X = self.X[item]
        if len(X.shape) == 2:
            X = np.repeat(X[None],3,axis=0).astype(np.float32)
        else:
            X = np.transpose(X, (2,0,1)).astype(np.float32)
        return X, self.y[item]

    def __len__(self):
        return len(self.X)

# 获取所有需要的数据
def getdataset(args,conf_dict):
    N_parties = conf_dict["N_parties"]
    N_samples_per_class = conf_dict["N_samples_per_class"]

    public_classes = conf_dict["public_classes"]
    private_classes = conf_dict["private_classes"]



    # 载入数据集
    if args.dataset == 'cifar':
        X_train_public, y_train_public, X_test_public, y_test_public = get_dataarray(args,dataset='cifar10')
        public_dataset = {"X": X_train_public, "y": y_train_public}

        X_train_private, y_train_private, X_test_private, y_test_private\
            = get_dataarray(args,dataset='cifar100')

    elif args.dataset == 'mnist':
        X_train_public, y_train_public, X_test_public, y_test_public = get_dataarray(args,dataset='mnist')
        public_dataset = {"X": X_train_public, "y": y_train_public}

        X_train_private, y_train_private, X_test_private, y_test_private \
            = get_dataarray(args,dataset='fmnist')

        y_train_private += len(public_classes)   #所有标签数值加上len（public_classes）这一常数
        y_test_private += len(public_classes)

    # only use those data whose y_labels belong to private_classes
    # 采样
    if args.iid:
        X_train_private, y_train_private \
            = generate_partial_data(X=X_train_private, y=y_train_private,       #取[private_class]中的data
                                    class_in_use=private_classes,
                                    verbose=True)

        X_test_private, y_test_private \
            = generate_partial_data(X=X_test_private, y=y_test_private,
                                    class_in_use=private_classes,
                                    verbose=True)

        # relabel the selected private data for future convenience
        for index, cls_ in enumerate(private_classes):
            y_train_private[y_train_private == cls_] = index + len(public_classes) #list的bool值索引
            y_test_private[y_test_private == cls_] = index + len(public_classes)
        del index, cls_

        print(pd.Series(y_train_private).value_counts())
        mod_private_classes = np.arange(len(private_classes)) + len(public_classes) #mod_private_classes=【0~len(private_classes)-1】每个元素加上常数len(public_classes)

        print("=" * 60)
        # generate private data
        private_data, total_private_data \
            = generate_bal_private_data(X_train_private, y_train_private,
                                        N_parties=N_parties,
                                        classes_in_use=mod_private_classes,
                                        N_samples_per_class=N_samples_per_class,
                                        data_overlap=False)
        print("=" * 60)
        X_tmp, y_tmp = generate_partial_data(X=X_test_private, y=y_test_private,    #主要是mod_private_classes：relabel the selected private data 
                                             class_in_use=mod_private_classes,
                                             verbose=True)
        private_test_data = {"X": X_tmp, "y": y_tmp}

    else:
        X_train_private, y_train_private \
            = generate_partial_data(X=X_train_private, y=y_train_private,
                                    class_in_use=private_classes,
                                    verbose=True)

        X_test_private, y_test_private\
            = generate_partial_data(X=X_test_private, y=y_test_private,
                                    class_in_use=private_classes,
                                    verbose=True)


        # relabel the selected private data for future convenience
        for index, cls_ in enumerate(private_classes):
            y_train_private[y_train_private == cls_] = index + len(public_classes)
            y_test_private[y_test_private == cls_] = index + len(public_classes)
        del index, cls_

        # print(pd.Series(y_train_private).value_counts())
        mod_private_classes = np.arange(len(private_classes)) + len(public_classes)

        print("=" * 60)
        # generate private data
        if args.dataset == 'cifar':
            users_index = cifar_noniid(y_train_private, N_parties)

            private_data, total_private_data \
                = get_sample_data(X_train_private, y_train_private,users_index,N_samples_per_class*18)
        else:
            users_index = mnist_noniid(y_train_private, N_parties)

            private_data, total_private_data \
                = get_sample_data(X_train_private, y_train_private,users_index,N_samples_per_class*6)

        print("=" * 60)
        X_tmp, y_tmp = generate_partial_data(X=X_test_private, y=y_test_private,
                                             class_in_use=mod_private_classes,
                                             verbose=True)
        private_test_data = {"X": X_tmp, "y": y_tmp}


    return [X_train_public, y_train_public,X_test_public, y_test_public],\
           [public_dataset,private_data, total_private_data,private_test_data ]

# 磁盘载入array数据
def get_dataarray(args,dataset):
    """ Returns train and test dataarray
    """
    data_dir = '../data/'+dataset
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True)
    elif dataset=='mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True)
        # sample training data amongst users FashionMNIST
    elif dataset=='fmnist':
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True)
    X_train, y_train= train_dataset.train_data,train_dataset.train_labels
    X_test, y_test = test_dataset.test_data,test_dataset.test_labels
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# 每类采样
def get_sample_data(X_train_private, y_train_private,users_index,N_samples_per_class=10):
    private_data,total_private_data = [],{'X':[],'y':[],'idx':[]}
    for k in users_index.keys():
        index = users_index[k]
        idx = np.random.choice(range(len(index)),N_samples_per_class)
        index_o = index[idx].astype(int)
        private_data.append({'X':X_train_private[index_o],'y':y_train_private[index_o],'idx':index_o})
        total_private_data['X'].extend(X_train_private[index_o].tolist())
        total_private_data['y'].extend(y_train_private[index_o].tolist())
        total_private_data['idx'].extend(index_o.tolist())

    return private_data, total_private_data

# 选择固定的label的数据
def generate_partial_data(X, y, class_in_use = None, verbose = False):

    if class_in_use is None:
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis = 0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)

    return X_incomplete, y_incomplete





def generate_bal_private_data(X, y, N_parties = 10, classes_in_use = range(11), 
                              N_samples_per_class = 20, data_overlap = False):
    """
    Input: 
    -- N_parties : int, number of collaboraters in this activity;
    -- classes_in_use: array or generator, the classes of EMNIST-letters dataset 
    (0 <= y <= 25) to be used as private data; 
    -- N_sample_per_class: int, the number of private data points of each class for each party
    
    return: 
    
    """
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties, 
                               replace = data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):           
            idx_tmp = idx[i * N_samples_per_class : (i + 1)*N_samples_per_class]
            if priv_data[i] is None:     #为每个party创建private数据表
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]
                
                
    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data

# 训练过程中，挑选随机挑选一部分public数据
def generate_alignment_data(X, y, N_alignment = 3000):
    
    index = np.random.choice(range(len(y)),N_alignment)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    else:
        X_alignment = X[index]
        y_alignment = y[index]
    alignment_data = {}
    alignment_data["idx"] = index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment
    
    return alignment_data


def generate_EMNIST_writer_based_data(X, y, writer_info, N_priv_data_min = 30, 
                                      N_parties = 5, classes_in_use = range(6)):
    
    # mask is a boolean array of the same shape as y
    # mask[i] = True if y[i] in classes_in_use
    mask = None
    mask = [y == i for i in classes_in_use]
    mask = np.any(mask, axis = 0)
    
    df_tmp = None
    df_tmp = pd.DataFrame({"writer_ids": writer_info, "is_in_use": mask})
    #print(df_tmp.head())
    groupped = df_tmp[df_tmp["is_in_use"]].groupby("writer_ids")
    
    # organize the input the data (X,y) by writer_ids.
    # That is, 
    # data_by_writer is a dictionary where the keys are writer_ids,
    # and the contents are the correcponding data. 
    # Notice that only data with labels in class_in_use are included.
    data_by_writer = {}
    writer_ids = []
    for wt_id, idx in groupped.groups.items():
        if len(idx) >= N_priv_data_min:  
            writer_ids.append(wt_id)
            data_by_writer[wt_id] = {"X": X[idx], "y": y[idx], 
                                     "idx": idx, "writer_id": wt_id}
            
    # each participant in the collaborative group is assigned data 
    # from a single writer.
    ids_to_use = np.random.choice(writer_ids, size = N_parties, replace = False)
    combined_idx = np.array([], dtype = np.int64)
    private_data = []
    for i in range(N_parties):
        id_tmp = ids_to_use[i]
        private_data.append(data_by_writer[id_tmp])
        combined_idx = np.r_[combined_idx, data_by_writer[id_tmp]["idx"]]
        del id_tmp
    
    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return private_data, total_priv_data


def generate_imbal_CIFAR_private_data(X, y, y_super, classes_per_party, N_parties,
                                      samples_per_class=7):

    priv_data = [None] * N_parties
    combined_idxs = []
    count = 0
    for subcls_list in classes_per_party:
        idxs_per_party = []
        for c in subcls_list:
            idxs = np.flatnonzero(y == c)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            idxs_per_party.append(idxs)
        idxs_per_party = np.hstack(idxs_per_party)
        combined_idxs.append(idxs_per_party)
        
        dict_to_add = {}
        dict_to_add["idx"] = idxs_per_party
        dict_to_add["X"] = X[idxs_per_party]
        #dict_to_add["y"] = y[idxs_per_party]
        #dict_to_add["y_super"] = y_super[idxs_per_party]
        dict_to_add["y"] = y_super[idxs_per_party]
        priv_data[count] = dict_to_add
        count += 1
    
    combined_idxs = np.hstack(combined_idxs)
    total_priv_data = {}
    total_priv_data["idx"] = combined_idxs
    total_priv_data["X"] = X[combined_idxs]
    #total_priv_data["y"] = y[combined_idxs]
    #total_priv_data["y_super"] = y_super[combined_idxs]
    total_priv_data["y"] = y_super[combined_idxs]
    return priv_data, total_priv_data

def main():
    '''显示图像'''
    import cv2
    xtrain,ytrain,xtest,ytest = get_dataarray(None,'fmnist')
    print(np.unique(ytrain),np.unique(ytest))
    print(xtrain.shape)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)
    for i in np.unique(ytrain):
        img = xtrain[ytrain==i][0]
        cv2.imwrite('./%d.png'%i,img)
if __name__ =='__main__':
    main()
