#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    conf_file = os.path.abspath("../conf/CIFAR_balance_conf.json")
    parser.add_argument('--conf', default=conf_file,
                        help='the config file for FedMD.'
                       )
    parser.add_argument('--use_pretrained_model', type=bool, default=False,
                        help="number of rounds of training")

    parser.add_argument('--gpu', type=float, default=None,
                        help='set gpu id')

    # 以下参数不修改
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")# cifar or mnist
    parser.add_argument('--iid', type=bool, default=False,
                        help='Default set to non-IID. Set to True for IID.')

    args = parser.parse_args()
    data_set ='mnist' if 'MNIST' in args.conf else 'cifar'
    args.dataset = data_set
    args.iid = False if 'imbalance' in args.conf else True
    print(args)
    return args

if __name__ == '__main__':
    args = args_parser()
    print('finish')