import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import pdb
from torch.autograd import Function

class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, lamda):
		ctx.lamda = lamda
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = (grad_output.neg() * ctx.lamda)
		return output, None


class encoder(nn.Module):
	def __init__(self, config):
		super(encoder, self).__init__()
		c_dim = 32
		self.conv1 = nn.Conv2d(config.channels, c_dim, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1)
		self.pool1 = nn.MaxPool2d(2, stride=2)

		self.conv3 = nn.Conv2d(c_dim, c_dim * 2, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(c_dim * 2, c_dim * 2, kernel_size=3, padding=1)
		self.pool2 = nn.MaxPool2d(2, stride=2)

		self.conv5 = nn.Conv2d(c_dim * 2, c_dim * 4, kernel_size=3, padding=1)
		self.conv6 = nn.Conv2d(c_dim * 4, c_dim * 4, kernel_size=3, padding=1)
		self.pool3 = nn.MaxPool2d(2, stride=2)

		# self.bn1 = nn.BatchNorm2d(c_dim)
		# self.bn2 = nn.BatchNorm2d(c_dim)
		# self.bn3 = nn.BatchNorm2d(c_dim*2)
		# self.bn4 = nn.BatchNorm2d(c_dim*2)
		# self.bn5 = nn.BatchNorm2d(c_dim*4)
		# self.bn6 = nn.BatchNorm2d(c_dim*4)
		# self.bn7 = nn.BatchNorm1d(config.deep_dim)

		self.flat_dim = (config.image_size//2//2//2) * (config.image_size//2//2//2) * c_dim * 4
		self.fc1 = nn.Linear(self.flat_dim, config.deep_dim)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)       
		
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool2(x)       
		
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = self.pool3(x)
		
		x = x.view(-1, self.flat_dim)
		x = F.relu(self.fc1(x))

		# x = F.relu(self.bn1(self.conv1(x)))
		# x = F.relu(self.bn2(self.conv2(x)))
		# x = self.pool1(x)       
		
		# x = F.relu(self.bn3(self.conv3(x)))
		# x = F.relu(self.bn4(self.conv4(x)))
		# x = self.pool2(x)       
		
		# x = F.relu(self.bn5(self.conv5(x)))
		# x = F.relu(self.bn6(self.conv6(x)))
		# x = self.pool3(x)
		
		# x = x.view(-1, self.flat_dim)
		# x = F.relu(self.bn7(self.fc1(x)))
		return x


class classifier(nn.Module):
	def __init__(self, config):
		super(classifier, self).__init__()
		self.l1 = nn.Linear(config.deep_dim, config.num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

	def forward(self, x):
		x = self.l1(x)
		return x


class discriminator(nn.Module):
	def __init__(self, config):
		super(discriminator, self).__init__()
		self.config = config
		self.l1 = nn.Linear(config.deep_dim, config.fd_h_dim)
		self.l2 = nn.Linear(config.fd_h_dim, config.fd_h_dim)
		out_dim = config.num_classes if config.alignment.lower() =='local' else 1
		self.l3 = nn.Linear(config.fd_h_dim, out_dim)

		# self.bn1 = nn.BatchNorm1d(config.fd_h_dim)
		# self.bn2 = nn.BatchNorm1d(config.fd_h_dim)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0.0, 0.02)

	def forward(self, x, alpha=1.0):
		if self.config.method.lower() == 'dann' :
			x = GradReverse.apply(x, alpha)
		x = F.leaky_relu(self.l1(x), 0.2)
		x = F.leaky_relu(self.l2(x), 0.2)
		x = self.l3(x)

		# x = F.leaky_relu(self.bn1(self.l1(x)), 0.2)
		# x = F.leaky_relu(self.bn2(self.l2(x)), 0.2)
		# x = self.l3(x)
		return torch.sigmoid(x)
