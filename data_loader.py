import torch
from torchvision import datasets
from torchvision import transforms
import os
import numpy as np
import pdb

def get_DB(db, img_transform=None, path=None, train=True):
	if db=='svhn':
		tr = transforms.Compose(img_transform)
		if train:
			return datasets.SVHN(path, split = 'train', download = True, transform=tr)
		return datasets.SVHN(path, split = 'test', download = True, transform=tr)

	elif db=='mnist':
		tr = transforms.Compose([transforms.Grayscale(3)] + img_transform)
		if train:
			return datasets.MNIST(path, train=True, download = True, transform=tr)
		return datasets.MNIST(path, train=False, download = True, transform=tr)

	elif db=='usps':
		tr = transforms.Compose(img_transform)
		if train:
			return datasets.USPS(path, train=True, download = True, transform=tr)
		return datasets.USPS(path, train=False, download = True, transform=tr)
		
	else:
		tr = transforms.Compose(img_transform)
		if train:
			return datasets.ImageFolder(root=os.path.join(path, db, 'trainset'),transform=tr)
		return datasets.ImageFolder(root=os.path.join(path, db, 'testset'),transform=tr)

def get_loader(config):
	
		# load data
		mean = np.array([0.5, 0.5, 0.5])
		std = np.array([0.5, 0.5, 0.5])
		
		img_transform = [transforms.Resize([config.image_size, config.image_size])]
		
		if config.channels == 1:
			img_transform += [transforms.Grayscale(1)]
			mean = np.array([0.5])
			std = np.array([0.5])

		img_transform += [transforms.ToTensor(), transforms.Normalize(mean, std)]

		source = get_DB(config.source, img_transform, config.db_path, train=True)		
		source_test = get_DB(config.source, img_transform, config.db_path, train=config.source_testset == 0)		
		target_train = get_DB(config.target, img_transform, config.db_path, train=True)
		target_test = get_DB(config.target, img_transform, config.db_path, train=config.target_testset == 0)

		source_loader = torch.utils.data.DataLoader(dataset=source,
													batch_size=config.batch_size,
													shuffle=True,
													num_workers=config.num_workers,
													drop_last=True)

		source_test_loader = torch.utils.data.DataLoader(dataset=source_test,
														batch_size=config.batch_size * 2,
														shuffle=False,
														num_workers=config.num_workers,
														drop_last=False)

		target_loader = torch.utils.data.DataLoader(dataset=target_train,
													 batch_size=config.batch_size ,
													 shuffle=True,
													 num_workers=config.num_workers,
													 drop_last=True)

		target_test_loader = torch.utils.data.DataLoader(dataset=target_test,
														 batch_size=config.batch_size * 2,
														 shuffle=False,
														 num_workers=config.num_workers,
														 drop_last=False)

		return source_loader, target_loader, source_test_loader, target_test_loader