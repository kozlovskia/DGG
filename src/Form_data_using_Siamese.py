# -*- coding: utf-8 -*-
from pathlib import Path
import os, argparse
import sys
import scipy
import timeit
import gzip
import torch

import numpy as np
from sys import stdout
import pickle as pkl

from torchvision import datasets, transforms

from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, Brach3, WineQuality, Banknote


def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

def cal_similar(data,K):
	data = data.cuda()
	N = data.shape[0]
	similar_m = []
	for idx in range(N):
		dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
		_, ind = dis.sort()
		select = np.random.permutation(100)
		select1 = select[0:K] + 1
		select1 = select1.tolist()
		temp = []
		temp.append(0)
		temp.extend(select1)
		similar_m.append(ind[temp].view(1,K+1).cpu())
		stdout.write('\r')    
		stdout.write("|index #{}".format(idx+1))
		stdout.flush()

	similar_m = torch.cat(similar_m,dim=0)

	return similar_m

def cal_err(data,index):
	data = data.cuda()
	index = index.cuda()
	N = data.shape[0]
	err = 0
	for idx in range(N):
		err = err + torch.sum((index[data[idx,:]] != index[data[idx,0]])*1.0).cpu()
		stdout.write('\r')    
		stdout.write("|index #{}".format(idx+1))
		stdout.flush()
	return err

def form_data(data, similar_m, save_dir):
	K = similar_m.shape[1]
	for idx in range(K):
		print(idx)
		data_s = data[similar_m[:,idx]]
		torch.save(to_numpy(data_s), save_dir / '{}_all_siamese.pkl'.format(idx))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', type=Path)
	parser.add_argument('--run_idx', type=int, default=0)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--dataset_dim', type=int, default=784)
	parser.add_argument('--channel_dims', type=int, nargs='+', default=[784, 500, 500, 2000])
	parser.add_argument('--n_clusters', default=10, type=int)
	parser.add_argument('--batch_size', default=1000, type=int)
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--n_epochs', default=300, type=int)
	parser.add_argument('--latent_dim', default=10, type=int)

	return parser.parse_args()



def main(args):
	data_save_dir = args.save_dir / 'data'
	data_save_dir.mkdir(parents=True, exist_ok=True)

	if args.dataset == 'mnist':
		dataset = CustomMNIST('./data')
	elif args.dataset == 'fmnist': 
		dataset = CustomFMNIST('./data')
	elif args.dataset == 'cifar10':
		dataset = CustomCIFAR10('./data')
	elif args.dataset == 'brach3':
		dataset = Brach3('./data/brach3-5klas.txt')
	elif args.dataset == 'winequality':
		dataset = WineQuality('./data/winequality-white.csv')
	elif args.dataset == 'banknote':
		dataset = Banknote('./data/data_banknote_authentication.txt')

	image_train = np.concatenate([el[0].view(-1, args.dataset_dim) for el in dataset], axis=0)
	label_train = np.array([el[1] for el in dataset])
	print(image_train.shape, label_train.shape)
	
	K = 40
	
	resume = 1

	if resume:
		similar_m = torch.load(args.save_dir / 'similar_m.pkl')
	else:
		similar_m = cal_similar(torch.from_numpy(image_train),K)
		torch.save(similar_m, args.save_dir / 'similar_m.pkl')
	print(similar_m.size())

	form_data(torch.from_numpy(image_train), similar_m, data_save_dir)
	torch.save(label_train, data_save_dir / 'label_all_siamese.pkl')


if __name__ == '__main__':
	main(parse_args())

